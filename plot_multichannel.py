'''
Created on Dec 02, 2016

@author: abhi
'''

import  struct
import sys
import numpy as np
from matplotlib import pyplot


def main():
    filename = sys.argv[1]
    channel1, channel2 = read(filename)
    plot(channel1, channel2)
    pyplot.show()


def read(filename):
    channel1 = []
    channel2 = []
    with open(filename, 'rb') as f:
        buff = f.read(512)
        while buff:
            ints = list(buff[i:i + 4] for i in xrange(0, len(buff), 4))
            for value in ints[::2]:
                channel1.append(parse_int(value))
            for value in ints[1::2]:
                channel2.append(parse_int(value))
            buff = f.read(512)
    return channel1, channel2


def parse_int(value):
    try:
        return struct.unpack("!L", value)[0]
    except struct.error:
        return -1


def plot(channel1, channel2):
    fig = pyplot.figure(figsize=(15, 6))

    ch1 = fig.add_subplot(1, 1, 1)
    ch1.plot(channel1, linewidth=1)

    ch2 = fig.add_subplot(1, 1, 1)
    ch2.plot(channel2, linewidth=1)


if __name__ == "__main__":
    main()
