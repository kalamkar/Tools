'''
Created on Dec 02, 2016

@author: abhi
'''

import  struct
import sys
import numpy as np
from scipy import signal
from matplotlib import pyplot


LOW_FREQ = 0.0005
HIGH_FREQ = 0.1

SAMPLING_RATE = 200
MAX_Y = 20000
BLINK_WINDOW = 20
MEDIAN_SIZE = BLINK_WINDOW * 8 + 1


def main():
    filename = sys.argv[1]
    channel1, channel2 = read(filename)
    channel1 = signal.detrend(channel1[500:3600])
    channel2 = signal.detrend(channel2[500:3600])

    figure = pyplot.figure(figsize=(15, 6))

    filtered1 = cleanup(channel1)
    plot(figure, filtered1, 'blue')
    plot(figure, baseline(filtered1), 'lightblue')

    # filtered2 = cleanup(channel2)
    # plot(figure, filtered2, 'green')
    # plot(figure, baseline(filtered2), 'lightgreen')

    pyplot.show()


def cleanup(data):
    b, a = signal.bessel(1, [rad_per_s(LOW_FREQ), rad_per_s(HIGH_FREQ)], btype='bandpass')
    return signal.lfilter(b, a, data)


def baseline(data):
    return signal.medfilt(data, kernel_size=MEDIAN_SIZE)
    # b, a = signal.bessel(1, [rad_per_s(LOW_FREQ)], btype='lowpass')
    # return signal.lfilter(b, a, data)


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
    chart.axis([0, len(channel), -MAX_Y, MAX_Y])


def rad_per_s(hertz):
    return 2 * 3.14159265 * hertz

if __name__ == "__main__":
    main()
