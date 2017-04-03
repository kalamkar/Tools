'''
Created on Dec 02, 2016

@author: abhi
'''

import csv
import sys
import numpy as np
from scipy import signal
from matplotlib import pyplot


SAMPLING_RATE = 51.2
START = int(SAMPLING_RATE * 15)  # Ignore first 15 seconds


def main():
    reader = csv.reader(open(sys.argv[1], 'rb'))
    columns = list(zip(*reader))

    figure = pyplot.figure(figsize=(15, 10))

    show_slope(columns, figure)
    # show_raw(columns, figure)

    pyplot.show()


def show_slope(columns, figure):
    raw1 = np.array(columns[0][START:-50]).astype(np.int)
    raw2 = np.array(columns[1][START:-50]).astype(np.int)

    channel11, channel12 = slopes(raw1)
    channel21, channel22 = slopes(raw2)

    slice_start = 7000
    slice_end = 9000 # len(raw1)

    channel1 = channel12[slice_start:slice_end]
    channel2 = channel22[slice_start:slice_end]

    raw1 = signal.detrend(raw1)[slice_start:slice_end]
    raw2 = signal.detrend(raw2)[slice_start:slice_end]

    plot(figure, 211, raw1, 'lightblue', window=len(raw1))
    plot(figure, 211, channel1, 'blue', window=len(channel1), twin=True)
    plot(figure, 212, raw2, 'lightgreen', window=len(raw2))
    plot(figure, 212, channel2, 'green', window=len(channel2), twin=True)


def slopes(data, slope_window_size=5, quality_window_size=5000):
    stage2 = [0] * slope_window_size
    quality = []
    for i in range(slope_window_size, len(data)):
        slope = get_slope(data[i - slope_window_size:i])
        quality.append(slope)
        if len(quality) > quality_window_size:
            quality = quality[1:]
        threshold = np.median(quality)
        stage2.append(slope if abs(slope) > threshold * 2 else 0)
    return [], stage2


def curvefit_median_diffdiff_data(data, cutoff):
    stage1 = []
    stage2 = []
    raw = [0] * 500
    straight = [0] * 20
    prev_median = 0
    prevdiff1 = 0
    prevdata = 0
    for i in range(0, len(data)):
        raw = raw[1:]
        raw.append(data[i])
        curve = get_curve(raw, degree=1)

        straight = straight[1:]
        straight.append(raw[len(raw) - 1] - np.polyval(curve, len(raw)))

        median = int(np.median(straight))
        stage1.append(median)

        diff1 = median - prev_median
        diff2 = diff1 - prevdiff1
        if abs(diff2) > cutoff:
            prevdata = median
        stage2.append(prevdata)

        prev_median = median
        prevdiff1 = diff1

    return stage1, stage2


def get_curve(data, degree=2):
    x = []
    y = []
    for i in range(0, len(data)):
        if i % 30 == 0:
            x.append(i)
            y.append(data[i])
    return np.polyfit(x, y, degree)


def get_slope(data):
    median_size = max(1, len(data) / 20) # 5%
    start = int(np.median(data[:median_size]))
    end = int(np.median(data[-median_size:]))
    return (end - start) / len(data)


def plot(figure, row_col, data, color, max_y=0, start=0,
         window=SAMPLING_RATE * 60, twin=False):
    chart = figure.add_subplot(row_col)
    if twin:
        chart = chart.twinx()
    chart.set_xticks(np.arange(0, len(data), window / 15))
    chart.plot(data, linewidth=1, color=color)
    chart.set_xbound([start, start + window])
    if max_y:
        chart.set_ybound([-max_y, max_y])


if __name__ == "__main__":
    main()
