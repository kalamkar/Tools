'''
Created on Dec 02, 2016

@author: abhi
'''

import csv
import sys
import math
import numpy as np
import time
from scipy import signal
from matplotlib import pyplot

SAMPLING_RATE = 51.2
START = int(SAMPLING_RATE * 15)  # Ignore first 15 seconds


def main():
    reader = csv.reader(open(sys.argv[1], 'rb'))
    columns = list(zip(*reader))

    figure = pyplot.figure(figsize=(15, 10))

    start = time.time()
    show_slope(columns, figure)
    print 'Processing took %d seconds' % (time.time() - start)

    pyplot.show()


def show_slope(columns, figure):
    raw1 = np.array(columns[0][START:-50]).astype(np.int)
    raw2 = np.array(columns[1][START:-50]).astype(np.int)

    channel1, drifts1, thresholds1 = slopes(raw1)
    channel2, drifts2, thresholds2  = slopes(raw2)

    slice_start = 6000
    slice_end = 8000 # len(raw1)

    channel1 = channel1[slice_start:slice_end]
    channel2 = channel2[slice_start:slice_end]
    drifts1 = drifts1[slice_start:slice_end]
    drifts2 = drifts2[slice_start:slice_end]
    thresholds1 = thresholds1[slice_start:slice_end]
    thresholds2 = thresholds2[slice_start:slice_end]

    raw1 = signal.detrend(raw1)
    raw2 = signal.detrend(raw2)
    raw1 = raw1[slice_start:slice_end]
    raw2 = raw2[slice_start:slice_end]

    plot(figure, 211, channel1, 'blue', window=len(channel1), max_y=1500)
    plot(figure, 211, raw1, 'lightblue', window=len(raw1), twin=True)
    plot(figure, 211, thresholds1, 'red', window=len(thresholds1), max_y=1500)
    # plot(figure, 211, drifts1, 'yellow', window=len(drifts1), twin=True, max_y=20, min_y=0)

    plot(figure, 212, channel2, 'green', window=len(channel2), max_y=1500)
    plot(figure, 212, raw2, 'lightgreen', window=len(raw2), twin=True)
    plot(figure, 212, thresholds2, 'red', window=len(thresholds2), max_y=1500)
    # plot(figure, 212, drifts2, 'yellow', window=len(drifts2), twin=True, max_y=20, min_y=0)


def slopes(data, slope_window_size=5):
    stage2 = [0] * slope_window_size

    thresholds = []
    slope_window = data[:slope_window_size - 1].tolist()

    drifts = [0] * len(data)
    drift_window = [0] * 1000
    current_drift = 0

    for i in range(slope_window_size, len(data)):
        slope_window = slope_window[1:]
        slope_window.append(data[i])
        slope = get_slope(slope_window)

        drift_window = drift_window[1:]
        drift_window.append(data[i])
        if i % 100 == 0:
            current_drift = get_slope(drift_window)
        threshold = drifts[i] = max(400, current_drift * 2)
        thresholds.append(threshold)

        stage2.append(slope if abs(slope) > abs(threshold) else 0)

    return stage2, drifts, thresholds


def get_slope(data):
    median_size = max(1, len(data) / 20)  # 5%
    start = int(np.median(data[:median_size]))
    end = int(np.median(data[-median_size:]))
    return (end - start) / len(data)


def plot(figure, row_col, data, color, max_y=0, min_y=-1, start=0, twin=False, window=SAMPLING_RATE * 60):
    chart = figure.add_subplot(row_col)
    if twin:
        chart = chart.twinx()
    chart.set_xticks(np.arange(0, len(data), window / 15))
    chart.plot(data, linewidth=1, color=color)
    chart.set_xbound([start, start + window])
    if max_y and min_y == -1:
        chart.set_ybound([-max_y, max_y])
    elif max_y:
        chart.set_ybound([min_y, max_y])


if __name__ == "__main__":
    main()