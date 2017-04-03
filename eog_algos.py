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

    slice_start = 0
    slice_end = len(raw1)

    channel1 = channel1[slice_start:slice_end]
    channel2 = channel2[slice_start:slice_end]
    drifts1 = drifts1[slice_start:slice_end]
    drifts2 = drifts2[slice_start:slice_end]
    thresholds1 = thresholds1[slice_start:slice_end]
    thresholds2 = thresholds2[slice_start:slice_end]

    # raw1 = signal.detrend(raw1)
    # raw2 = signal.detrend(raw2)
    raw1 = raw1[slice_start:slice_end]
    raw2 = raw2[slice_start:slice_end]

    # plot(figure, 211, channel1, 'blue', window=len(channel1), max_y=1500)
    plot(figure, 211, raw1, 'lightblue', window=len(raw1))
    # plot(figure, 211, thresholds1, 'red', window=len(thresholds1), max_y=1500)
    plot(figure, 211, drifts1, 'yellow', window=len(drifts1), twin=True, max_y=250, min_y=0)

    # plot(figure, 212, channel2, 'green', window=len(channel2), max_y=1500)
    plot(figure, 212, raw2, 'lightgreen', window=len(raw2))
    # plot(figure, 212, thresholds2, 'red', window=len(thresholds2), max_y=1500)
    plot(figure, 212, drifts2, 'yellow', window=len(drifts2), twin=True, max_y=250, min_y=0)


def slopes(data, slope_window_size=5, quality_window_size=50):
    stage2 = [0] * slope_window_size

    thresholds = []
    quality_window = []
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
        if i % 500 == 0:
            current_drift = get_slope(drift_window)
        percentile = max(90, 100 - math.sqrt(abs(current_drift)))
        drifts[i] = abs(current_drift)

        quality_window.append(slope)
        if len(quality_window) > quality_window_size:
            quality_window = quality_window[1:]
        threshold = abs(int(np.percentile(quality_window, percentile)) * 2)

        stage2.append(slope if abs(slope) > threshold else 0)
        thresholds.append(threshold)

    return stage2, drifts, thresholds


def curvefit_median_diffdiff_data(data, cutoff, curvefit_size=500, median_size=20):
    stage1 = []
    stage2 = []
    raw = [0] * curvefit_size
    straight = [0] * median_size
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

    return stage2, stage1


def get_curve(data, degree=2):
    x = []
    y = []
    for i in range(0, len(data)):
        if i % 30 == 0:
            x.append(i)
            y.append(data[i])
    return np.polyfit(x, y, degree)


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
