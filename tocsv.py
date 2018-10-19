
def tocsv(filename, output):
    with open(filename, 'r') as inp, open(output, 'w') as out:
        for line in inp:
            if line.startswith("="):
                out.write(line)
            else:
                prodId, userId, score, timestamp = line.split('    ')
                out.write("{},{},{},{}".format(
                    prodId.split(":")[1].strip(),
                    userId.split(":")[1].strip(),
                    score.split(":")[1].strip(),
                    timestamp.split(":")[1].strip()
                ))
                out.write("\n")


if __name__ == "__main__":
    tocsv("data/reviews_for_each_user.txt", "data/reviews_for_each_user.csv")


"""
  def __create_dataset(self, filename):
        cur_user_id = 0
        cur_movie_id = 0

        with open(filename, 'r') as f:
            for line in f:
                movieId, userId, rating, timestamp = line.split(',')

                if userId in self.userToId:
                    self.userToId[userId] = cur_user_id
                    self.idToUser[cur_user_id] = userId
                    cur_user_id += 1

                if movieId in self.movieToId:
                    self.movieToId[movieId] = cur_movie_id
                    self.idToMovie[cur_movie_id] = movieId
                    cur_movie_id += 1

                cur_data = DataRow(self.userToId[userId], self.movieToId[movieId], float(rating), int(timestamp))

                self.dataset.append(cur_data)
"""