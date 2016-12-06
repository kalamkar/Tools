'''
Created on Dec 02, 2016

@author: abhi
'''

import  struct
import sys
import numpy as np
from scipy import signal
from matplotlib import pyplot

SAMPLING_RATE = 200
MAX_Y = 200000
MAX_X = 3600
MIN_X = 50
FACTOR = 50


def main():
    filename = sys.argv[1]
    channel1, channel2 = read(filename)
    channel1 = signal.detrend(channel1)
    channel2 = signal.detrend(channel2)

    figure = pyplot.figure(figsize=(15, 6))

    # plot(figure, channel1, 'b')
    # plot(figure, np.array(bessel(channel1)) * FACTOR, 'lightblue')

    plot(figure, channel2, 'g')
    plot(figure, np.array(bessel(channel2)) * FACTOR, 'lightgreen')

    pyplot.show()


def bessel(data):
    b, a = signal.bessel(1, [rad_per_s(0.5), rad_per_s(840)], btype='bandpass')
    return signal.lfilter(b, a, data)


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


def plot(figure, channel, color):
    chart = figure.add_subplot(1, 1, 1)
    chart.set_xticks(np.arange(0, len(channel), SAMPLING_RATE))
    chart.plot(channel, linewidth=1, color=color)
    chart.axis([MIN_X, MAX_X, -MAX_Y, MAX_Y])


def rad_per_s(hertz):
    return 2 * 3.14159265 * hertz

if __name__ == "__main__":
    main()
