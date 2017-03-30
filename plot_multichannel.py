'''
Created on Dec 02, 2016

@author: abhi
'''

import csv
import struct
import sys
import numpy as np
from scipy import signal
from matplotlib import pyplot


LOW_FREQ = 0.0625
HIGH_FREQ = 1.024

SAMPLING_RATE = 51.2
MAX_Y = 20000
BLINK_WINDOW = 20
MEDIAN_SIZE = BLINK_WINDOW * 16 + 1


def main():

    reader = csv.reader(open(sys.argv[1], 'rb'))
    columns = list(zip(*reader))

    # show_raw(columns)
    # show_filtered(columns)
    show_raw_filtered(columns)
    # show_bandpass(columns)


def show_bandpass(columns):
    channel1 = signal.detrend(columns[0][1:])
    channel2 = signal.detrend(columns[1][1:])

    figure = pyplot.figure(figsize=(15, 6))

    filtered1 = cleanup(channel1)
    plot(figure, filtered1, 'blue', MAX_Y)
    # plot(figure, baseline(filtered1), 'lightblue', MAX_Y)

    filtered2 = cleanup(channel2)
    plot(figure, filtered2, 'green', MAX_Y)
    # plot(figure, baseline(filtered2), 'lightgreen', MAX_Y)

    pyplot.show()


def show_filtered(columns):
    channel1 = columns[2][1:]
    channel2 = columns[3][1:]

    figure = pyplot.figure(figsize=(15, 6))

    plot(figure, channel1, 'blue', 20000)
    plot(figure, channel2, 'green', 20000)

    pyplot.show()


def show_raw_filtered(columns):
    channel1 = signal.detrend(columns[0][1:])
    channel2 = signal.detrend(columns[1][1:])

    figure = pyplot.figure(figsize=(15, 6))

    plot(figure, columns[2][1:], 'blue', 20000)
    plot(figure, columns[3][1:], 'green', 20000)

    plot(figure, channel1, 'lightblue', 200000)
    plot(figure, channel2, 'lightgreen', 200000)

    pyplot.show()


def show_raw(columns):
    channel1 = signal.detrend(columns[0][1:])
    channel2 = signal.detrend(columns[1][1:])

    figure = pyplot.figure(figsize=(15, 6))

    plot(figure, channel1, 'blue', 20000)
    plot(figure, channel2, 'green', 20000)

    pyplot.show()


def cleanup(data):
    b, a = bessel_bandpass(LOW_FREQ, HIGH_FREQ, SAMPLING_RATE)
    return signal.lfilter(b, a, data)


def baseline(data):
    return signal.medfilt(data, kernel_size=MEDIAN_SIZE)
    # b, a = signal.bessel(1, [rad_per_s(LOW_FREQ)], btype='lowpass')
    # return signal.lfilter(b, a, data)


def parse_int(value):
    try:
        return struct.unpack("!L", value)[0]
    except struct.error:
        return -1


def plot(figure, channel, color, maxY=-1):
    chart = figure.add_subplot(1, 1, 1)
    chart.set_xticks(np.arange(0, len(channel), SAMPLING_RATE * 5))
    chart.plot(channel, linewidth=1, color=color)
    pyplot.xlim([0, SAMPLING_RATE * 60])
    if maxY > 0:
        pyplot.ylim([-maxY, maxY])


def rad_per_s(hertz):
    return 2.0 * np.pi * hertz


def butter_bandpass(lowcut, highcut, fs, order=1):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    return b, a

def bessel_bandpass(lowcut, highcut, fs, order=1):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.bessel(order, [low, high], btype='band')
    return b, a


if __name__ == "__main__":
    main()
