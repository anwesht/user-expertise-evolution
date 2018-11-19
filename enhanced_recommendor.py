# Author: Anwesh Tuladhar <anwesh.tuladhar@gmail.com>
# Website: https://anwesht.github.io/
__author__ = 'Anwesh Tuladhar'

from typing import NamedTuple
from operator import attrgetter
import pprint

import tensorflow as tf
import numpy as np

import os
import math
import time


DATA_FILE = 'data/reviews_for_each_user.csv'
LOGS_PATH = 'logs'
# --------------
#  Debug params
# --------------
DEBUG = True
# DEBUG = False
MAX_USERS = 50
# Enable eager execution to peek at values of tensors
# if DEBUG:
#     tf.enable_eager_execution()


# Model Arguments
model_args = {
    'num_latent_factors': 5,
    'num_iters': 10000,
    'reg_factor': 0.01,
    'learning_rate': 0.01
}


def dprint(s):
    if DEBUG:
        print(s)


class DataRow(NamedTuple):
    """
    Class to represent each row of data for easy access.
    """
    user: int
    movie: int
    rating: float
    timestamp: int
    experience: int


class DataLoader:
    def __init__(self, filename, num_experience=5):
        """
        Reads the data file and create lookup tables.
        :param filename: str : the path to the file
        """
        self.filename = filename
        self.num_experience = num_experience

        self.userToId = dict()
        self.idToUser = dict()

        self.movieToId = dict()
        self.idToMovie = dict()

        self.ratingsPerUser = dict()  # key: userid, value: list of DataRow
        self.validationRatingsPerUser = dict()
        self.testRatingsPerUser = dict()

        self.numValidationPerUser = 1
        self.numTestPerUser = 1

        self.num_users = 0
        self.num_movies = 0

        self.__create_dataset(filename)

        # Incorporate experience in model
        self.num_rows = self.num_users * self.num_experience
        self.num_columns = self.num_movies

    @staticmethod
    def split_list(src_list, num_to_remove=0, ):
        valid_indices = np.random.choice(a=len(src_list),
                                         size=num_to_remove,
                                         replace=False)
        dest_list = list()
        for i in valid_indices:
            dest_list.append(src_list.pop(i))

        return dest_list

    def __create_dataset(self, filename):
        """
        Read data from file, create lookup maps, create per user dataset (training, validation and test)
        with experience level.
        :param filename: file with raw data
        :return:
        """
        def map_experience(cur_datarow):
            for i in range(self.num_experience):
                # time_delta and min_time are captured variables.
                if (min_time + i*time_delta) <= cur_datarow.timestamp <= (min_time + (i+1)*time_delta):
                    # XXX: creating a new namedtuple because it is immutable.
                    # Use dataclasses if python version >= 3.7
                    return DataRow(
                        cur_datarow.user,
                        cur_datarow.movie,
                        cur_datarow.rating,
                        cur_datarow.timestamp,
                        i + 1,
                    )

            raise Exception("Timestamp of review not within the time frame.")


        cur_user_id = 0
        cur_movie_id = 0
        movie_index = 0

        cur_user_ratings = list()

        with open(filename, 'r') as f:
            for line in f:
                if line.startswith("="):
                    # do sorting and time range calculations
                    cur_user_ratings.sort(key=attrgetter('timestamp'))
                    min_time = cur_user_ratings[0].timestamp
                    max_time = cur_user_ratings[-1].timestamp

                    time_delta = (max_time - min_time) // self.num_experience

                    cur_user_ratings = list(map(map_experience, cur_user_ratings))

                    dprint("len of cur user {} ratings 1. : {}".format(cur_user_id, len(cur_user_ratings)))

                    # Separate validation set:
                    self.validationRatingsPerUser[cur_user_id] = self.split_list(cur_user_ratings,
                                                                                 self.numValidationPerUser)
                    dprint("len of cur user {} ratings: 2. {}".format(cur_user_id, len(cur_user_ratings)))

                    # Separate test set:
                    self.testRatingsPerUser[cur_user_id] = self.split_list(cur_user_ratings,
                                                                           self.numTestPerUser)
                    dprint("len of cur user {} ratings: 3. {}".format(cur_user_id, len(cur_user_ratings)))

                    # Training set:
                    self.ratingsPerUser[cur_user_id] = cur_user_ratings

                    # Update user maps
                    # Doing it here instead of like movieId because of the format of the data file.
                    self.userToId[userId] = cur_user_id
                    self.idToUser[cur_user_id] = userId
                    cur_user_id += 1

                    cur_user_ratings = list()

                else:
                    movieId, userId, rating, timestamp = line.split(',')

                    # Get current movie id.
                    if movieId not in self.movieToId:
                        self.movieToId[movieId] = movie_index
                        self.idToMovie[movie_index] = movieId
                        movie_index += 1
                    cur_movie_id = self.movieToId[movieId]

                    cur_data = DataRow(
                        cur_user_id,
                        cur_movie_id,
                        float(rating),
                        int(timestamp),
                        0  # Default experience = 0 -> updated after loading all data for user.
                    )

                    cur_user_ratings.append(cur_data)

                # For debugging purposes, limit number of users to max_user
                if DEBUG and cur_user_id == MAX_USERS:
                    break

        self.num_users = len(self.userToId)
        self.num_movies = movie_index

        dprint("Num Users: {}, Num Movies: {}".format(self.num_users, self.num_movies))


def train_model(d):
    num_rows = d.num_rows
    num_columns = d.num_columns

    # Create one giant matrix by accounting for user and there corresponding experience as follows:
    # index into user column = user_id + num_users * (experience - 1)
    users = []
    movies = []
    ratings = []
    for user_rating_list in d.ratingsPerUser.values():
        for datarow in user_rating_list:
            user_exp_ind = datarow.user + d.num_users * (datarow.experience - 1)
            users.append(user_exp_ind)
            movies.append(datarow.movie)
            ratings.append(datarow.rating)

    users_np = np.array(users)
    movies_np = np.array(movies)
    ratings_np = np.array(ratings)

    dprint("Shapes => users_np: {}, movies_np: {}, ratings_np: {}".format(users_np.shape, movies_np.shape, ratings_np.shape))

    # Matrix Factorization
    # initialize matrix with random values from a normal distribution
    user_factors = tf.Variable(initial_value=tf.truncated_normal(
        shape=[num_rows, model_args['num_latent_factors']],
        stddev=0.02, mean=3),
        name='user_factors')

    movie_factors = tf.Variable(tf.truncated_normal(
        shape=[model_args['num_latent_factors'], num_columns],
        stddev=0.02, mean=0,
        name='movie_factors'
    ))

    pred_matrix = tf.matmul(user_factors, movie_factors, name="pred_matrix")

    # Gather all the values from the resulting matrix for which we have ratings in the training data.
    pred_matrix_flat = tf.reshape(pred_matrix, [-1])
    indices = users_np * tf.shape(pred_matrix)[1] + movies_np

    dprint("params shape: {}".format(pred_matrix_flat.get_shape()))
    dprint("params: \n{}".format(pred_matrix_flat))
    dprint("indices shape: {}".format(indices.get_shape()))
    dprint("indices: \n{}".format(indices))

    pred_matrix_known_values = tf.gather(
        params=pred_matrix_flat,  # flatten matrix into 1-D array
        indices=indices,  # 2-D matrix to 1-D representation
        name="extract_training_ratings"
    )

    print("pred-matrix-known-values: shape: {}".format(pred_matrix_known_values.get_shape()))
    print("length of ratings {}".format(len(ratings)))

    # Find the difference between the actual values and the calculated values. i.e. error.
    diff = tf.subtract(pred_matrix_known_values, ratings_np, name="training_error")

    dprint("pred_matrix_known_values: \n{}".format(pred_matrix_known_values))
    dprint("ratings_np: \n{}".format(ratings_np))
    dprint("diff: \n{}".format(diff))

    # Regularization
    user_factor_splits = tf.split(user_factors, num_or_size_splits=d.num_experience, axis=0)
    user_factors_diff = []
    for i in range(len(user_factor_splits) - 1):
        user_factors_diff.append(tf.subtract(user_factor_splits[i], user_factor_splits[i+1]))

    user_factors_norm = tf.norm(tf.concat(user_factors_diff, axis=0), ord=2)

    total_norm = tf.add(user_factors_norm, tf.norm(tf.reshape(movie_factors, [-1])))

    regularization = tf.multiply(tf.constant(model_args['reg_factor'], name='lambda'), total_norm)

    squared_error = tf.square(diff, name="squared_error")

    cost = tf.reduce_sum(tf.add(regularization, squared_error), name="sum_squared_error")

    # Update: Trying to ensure non-negative values for user_factors and movie factors
    clip_user = user_factors.assign(tf.maximum(tf.zeros_like(user_factors), user_factors))
    clip_movie = movie_factors.assign(tf.maximum(tf.zeros_like(movie_factors), movie_factors))
    clip = tf.group(clip_user, clip_movie)

    average_cost = tf.div(cost, len(ratings_np)*2, "average_cost")

    # Training operation
    train_step = tf.train.AdagradOptimizer(learning_rate=model_args['learning_rate']).minimize(average_cost)

    # Calculate mean square error
    rmse = tf.sqrt(tf.reduce_sum(tf.square(diff)) / len(ratings_np))

    # Calculate Test Error
    users_test = []
    movies_test = []
    ratings_test = []
    for user_rating_list in d.validationRatingsPerUser.values():
        for datarow in user_rating_list:
            user_exp_ind = datarow.user + d.num_users * (datarow.experience - 1)
            users_test.append(user_exp_ind)
            movies_test.append(datarow.movie)
            ratings_test.append(datarow.rating)

    users_test_np = np.array(users_test)
    movies_test_np = np.array(movies_test)
    ratings_test_np = np.array(ratings_test)

    pred_matrix_test_values = tf.gather(pred_matrix_flat, users_test_np * tf.shape(pred_matrix)[1] + movies_test_np,
                       name='extracting_user_rate_test')

    diff_op_test = tf.subtract(pred_matrix_test_values, ratings_test_np, name='test_diff')

    rmse_test = tf.sqrt(tf.reduce_sum(tf.square(diff_op_test)) / len(ratings_test_np))

    # Create summary to monitor average cost tensor
    tf.summary.scalar("Average Training Error", average_cost)
    # Create summary to monitor mean squared error
    tf.summary.scalar("RMSE: Training", rmse)
    # Create summary to monitor mean squared error on test data.
    tf.summary.scalar("RMSE: Testing", rmse_test)
    # Merge all summaries into a single op
    merged_summaries_op = tf.summary.merge_all()

    # Run Training loop
    final_checkpoint = run_training_loop(average_cost, clip, cost, rmse_test, d, movie_factors, pred_matrix, rmse,
                                         train_step, user_factors, merged_summaries_op)
    # final_checkpoint = 'checkpoints/rec_train_1540158578'

    return final_checkpoint


def run_training_loop(average_cost, clip, cost, rmse_test, d, movie_factors, pred_matrix, rmse, train_step,
                      user_factors, merged_summaries_op):
    # Create a saver
    if not os.path.exists("checkpoints"):
        os.mkdir("checkpoints")
    saver = tf.train.Saver(max_to_keep=1000)

    init = tf.global_variables_initializer()

    losses_for_plot = []

    with tf.Session() as sess:
        sess.run(init)

        # op to write logs to Tensorboard
        summary_writer = tf.summary.FileWriter(LOGS_PATH, graph=tf.get_default_graph())

        for i in range(model_args['num_iters']):
            _, summary = sess.run([train_step, merged_summaries_op])
            # Write logs at every iteration
            summary_writer.add_summary(summary, i)

            # update:
            sess.run(clip)

            if i % 100 == 0:
                cost_out, pred_out, rmse_out = sess.run([cost, average_cost, rmse])
                print("iter: {} -> rmse: {}, cost: {}, average cost: {}".format(i, rmse_out, cost_out, pred_out))
                print("Test RMSE: {}".format(sess.run(rmse_test)))

                losses_for_plot.append(rmse_out)

            if i % 1000 == 0:
                timestamp = str(math.trunc(time.time()))
                learnt_user_factors = sess.run(user_factors)
                learnt_movie_factors = sess.run(movie_factors)
                print("saving pred matrix: \n {}".format(sess.run(pred_matrix)))
                saved_file = saver.save(sess, 'checkpoints/rec_train_' + timestamp)
                print("Saved file: " + saved_file)

                # example  DataRow(user=0, movie=85, rating=4.0, timestamp=947030400, experience=5)]
                d_test = d.ratingsPerUser[0][-1]

                user_index = d_test.user + d.num_users * (d_test.experience - 1)
                rhat = tf.gather(tf.gather(pred_matrix, user_index), d_test.movie)
                print("rating for user " + str(d_test.user) + " for item " + str(d_test.movie)
                      + " is " + str(d_test.rating) + " and our prediction is: " + str(
                    sess.run(rhat)))

    # Return final saved checkpoint
    return saved_file


def run_test_loop(d, checkpoint_file):
    saved_checkpoint = checkpoint_file
    saved_meta = checkpoint_file + '.meta'

    with tf.Session() as testsess:
        new_saver = tf.train.import_meta_graph(saved_meta)
        new_saver.restore(testsess, saved_checkpoint)

        graph = tf.get_default_graph()

        pred_matrix = graph.get_tensor_by_name("pred_matrix:0")

        print("Loaded pred_matrix: \n{}".format(pred_matrix))

        for user_rating_list in d.testRatingsPerUser.values():
            for d_test in user_rating_list:
                user_index = d_test.user + d.num_users * (d_test.experience - 1)
                pred = tf.gather(tf.gather(pred_matrix, user_index), d_test.movie)
                print("Actual Rating: {}, Prediction at experience level {}: {}".format(d_test.rating,
                                                                                        d_test.experience,
                                                                                        str(testsess.run(pred))))


def main():
    d = DataLoader(DATA_FILE)
    pprint.pprint(d.ratingsPerUser[0], depth=2, width=50)
    final_checkpoint = train_model(d)
    # final_checkpoint = 'checkpoints_presentation_100users_0.01lr_0.01reg/rec_train_1540221661'
    run_test_loop(d, final_checkpoint)


if __name__ == '__main__':
    main()





