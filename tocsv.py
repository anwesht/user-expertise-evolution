# Author: Anwesh Tuladhar <anwesh.tuladhar@gmail.com>
# Website: https://anwesht.github.io/
__author__ = 'Anwesh Tuladhar'


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
