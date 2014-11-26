import random
import sys

def process(source, dest):
    fo = open(dest, 'w')
    proc_line = []
    with open(source) as f:
        for line in f:
            words = line.split()
            # get random length of line and keep adding to  line till its acceptable length
            if len(proc_line) < random.choice([15, 40, 60, 100, 120]):
                proc_line += words
                continue
            else:   proc_line += (words)
            # write line to file.
            fo.write(' '.join(proc_line))
            fo.write('\n')
            proc_line = []

    fo.close()

def main(argv):
    process(argv[0], argv[1])

if __name__ == "__main__":
    main(sys.argv[1:])