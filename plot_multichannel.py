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
DOWN_SAMPLE_FACTOR = 30
MAX_Y = 20000
BLINK_WINDOW = 20
MEDIAN_SIZE = BLINK_WINDOW * 16 + 1


def main():
    reader = csv.reader(open(sys.argv[1], 'rb'))
    columns = list(zip(*reader))

    figure = pyplot.figure(figsize=(15, 6))

    show_algo(columns, figure)
    # show_raw(columns, figure)
    # show_filtered(columns, figure)
    # show_raw_filtered(columns, figure)

    pyplot.show()


def show_algo(columns, figure):
    channel1 = remove_drift(columns[0][1:])
    channel2 = remove_drift(columns[1][1:])

    # channel1 = remove_drift(columns[0][7000:9500])
    # channel2 = remove_drift(columns[1][7000:9500])

    # channel1 = signal.detrend(columns[0][7000:9500])
    # channel2 = signal.detrend(columns[1][7000:9500])

    # channel1 = median(channel1)
    # channel2 = median(channel2)

    channel1 = signal.medfilt(channel1, kernel_size=21)
    channel2 = signal.medfilt(channel2, kernel_size=21)

    # plot(figure, channel1, 'lightblue', 10000)
    plot(figure, channel2, 'lightgreen', 10000)

    # channel1 = diffdiff(channel1)
    # channel2 = diffdiff(channel2)

    # channel1 = diffdiff_filter(channel1, 150)
    channel2 = diffdiff_filter(channel2, 150)

    # plot(figure, channel1, 'blue', 10000)
    plot(figure, channel2, 'green', 10000)


def show_filtered(columns, figure):
    channel1 = columns[2][1:]
    channel2 = columns[3][1:]

    plot(figure, channel1, 'blue', 20000)
    plot(figure, channel2, 'green', 20000)


def show_raw_filtered(columns, figure):
    channel1 = signal.detrend(columns[0][1:])
    channel2 = signal.detrend(columns[1][1:])

    plot(figure, columns[2][1:], 'blue', 20000)
    plot(figure, columns[3][1:], 'green', 20000)

    plot(figure, channel1, 'lightblue', 200000)
    plot(figure, channel2, 'lightgreen', 200000)


def show_raw(columns, figure):
    channel1 = signal.detrend(columns[0][1:])
    channel2 = signal.detrend(columns[1][1:])

    plot(figure, channel1, 'blue', 20000)
    plot(figure, channel2, 'green', 20000)


def baseline(data):
    return signal.medfilt(data, kernel_size=MEDIAN_SIZE)


def diffdiff(data):
    lastdiff1 = data[1] - data[0]
    filtered = [0, lastdiff1]
    for i in range(2, len(data)):
        diff1 = data[i] - data[i-1]
        diff2 = diff1 - lastdiff1
        filtered.append(diff2)
        lastdiff1 = diff1
    return filtered


def diffdiff_filter(data, cutoff):
    lastdiff1 = data[1] - data[0]
    filtered = [0, lastdiff1]
    lastdata = data[0]
    for i in range(2, len(data)):
        diff1 = data[i] - data[i-1]
        diff2 = diff1 - lastdiff1
        if abs(diff2) > cutoff:
            lastdata = data[i]
        filtered.append(lastdata)
        lastdiff1 = diff1
    return filtered

def diff(data):
    filtered = [0]
    for i in range(1, len(data)):
        filtered.append(data[i] - data[i-1])
    return filtered


def median(data):
    filtered = [0]
    for i in range(20, len(data)):
        filtered.append(int(np.median(data[i-20:i])))
    return filtered


def remove_drift(data):
    filtered = []
    for i in range(0, len(data) - 500, 500):
        curve = get_curve(data[i:i+500])
        chunk = []
        for j in range(i, i+500):
            value = int(data[j]) - np.polyval(curve, i + j)
            chunk.append(value)
        filtered.extend(signal.detrend(chunk))

    return filtered


def get_curve(data):
    x = []
    y = []
    for i in range(0, len(data)):
        if i % DOWN_SAMPLE_FACTOR == 0:
            x.append(i)
            y.append(int(data[i]))
    return np.polyfit(x, y, 2)


def plot(figure, channel, color, maxY=-1):
    chart = figure.add_subplot(1, 1, 1)
    chart.set_xticks(np.arange(0, len(channel), SAMPLING_RATE * 5))
    chart.plot(channel, linewidth=1, color=color)
    if maxY > 0:
        chart.axis([0, SAMPLING_RATE * 60, -maxY, maxY])
    else:
        chart.set_xlim([0, SAMPLING_RATE * 60])


def rad_per_s(hertz):
    return 2.0 * np.pi * hertz


if __name__ == "__main__":
    main()
