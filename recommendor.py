from typing import NamedTuple
from operator import attrgetter
import pprint

import tensorflow as tf
import numpy as np


DATA_FILE = 'data/reviews_for_each_user.csv'

# Debug params
DEBUG = True
MAX_USERS = 10


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

                    print("len of cur user {} ratings 1. : {}".format(cur_user_id, len(cur_user_ratings)))
                    # Separate validation set:
                    self.validationRatingsPerUser[cur_user_id] = self.split_list(cur_user_ratings,
                                                                                 self.numValidationPerUser)
                    print("len of cur user {} ratings: 2. {}".format(cur_user_id, len(cur_user_ratings)))
                    # Separate test set:
                    self.testRatingsPerUser[cur_user_id] = self.split_list(cur_user_ratings,
                                                                           self.numTestPerUser)
                    print("len of cur user {} ratings: 3. {}".format(cur_user_id, len(cur_user_ratings)))
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


# Model Arguments
model_args = {
    'num_latent_factors' : 5,
    'num_iters' : 5,
    'reg_factor' : 1,

}
def main():
    d = DataLoader(DATA_FILE)
    pprint.pprint(d.ratingsPerUser[0], depth=2, width=50)


if __name__ == '__main__':
    main()





