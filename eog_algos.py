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
MIN_BLINK_HEIGHT = 10000


def main():
    reader = csv.reader(open(sys.argv[1], 'rb'))
    columns = list(zip(*reader))

    figure = pyplot.figure(figsize=(15, 10))

    start = time.time()
    show_slope(columns, figure)
    print 'Processing took %d seconds' % (time.time() - start)

    pyplot.show()


def show_slope(columns, figure):
    col = np.array(columns[6][START:-50]).astype(np.int)
    row = np.array(columns[7][START:-50]).astype(np.int)

    raw1 = np.array(columns[0][START:-50]).astype(np.int)
    raw2 = np.array(columns[1][START:-50]).astype(np.int)

    filtered1, [markers11, markers12], blink_points1 = filter_drift(raw1)
    filtered2, [markers21, markers22], blink_points2 = filter_drift(raw2)

    slice_start = 10
    slice_end = len(raw1) - 10

    blink_points1 = [i - slice_start if slice_start <= i < slice_end else 0 for i in blink_points1]
    blink_points2 = [i - slice_start if slice_start <= i < slice_end else 0 for i in blink_points2]
    blink_values1 = [filtered1[i] for i in blink_points1]
    blink_values2 = [filtered2[i] for i in blink_points2]

    # markers11 = markers21 = []
    # markers11 = markers11[slice_start:slice_end]
    # markers21 = markers21[slice_start:slice_end]

    # markers12 = markers22 = []
    # markers12 = markers12[slice_start:slice_end]
    # markers22 = markers22[slice_start:slice_end]

    # channel1 = channel2 = []
    channel1 = filtered1[slice_start:slice_end]
    channel2 = filtered2[slice_start:slice_end]

    # raw1 = signal.detrend(raw1)
    # raw2 = signal.detrend(raw2)
    raw1 = raw1[slice_start:slice_end]
    raw2 = raw2[slice_start:slice_end]

    row = [4 - x for x in row[slice_start:slice_end]]
    col = [4 - x for x in col[slice_start:slice_end]]

    plot(figure, 211, channel1, 'blue', window=len(channel1))
    plot(figure, 211, markers11, 'orange', window=len(markers11))
    plot(figure, 211, markers12, 'yellow', window=len(markers12))
    # plot(figure, 211, raw1, 'lightblue', window=len(raw1), twin=True)
    plot(figure, 211, col, 'lightblue', window=len(col), twin=True, min_y=-2, max_y=5)
    plot(figure, 211, blink_values1, 'orange', x=blink_points1, window=len(channel1))

    plot(figure, 212, channel2, 'green', window=len(channel2))
    plot(figure, 212, markers21, 'orange', window=len(markers21))
    plot(figure, 212, markers22, 'yellow', window=len(markers22))
    # plot(figure, 212, raw2, 'lightgreen', window=len(raw2), twin=True)
    plot(figure, 212, row, 'lightgreen', window=len(row), twin=True, min_y=-2, max_y=5)
    plot(figure, 212, blink_values2, 'orange', x=blink_points2, window=len(channel2))


def filter_drift(data, drift_window_size=500, update_interval=500, blink_window_size=30):
    filtered = []
    blinks = []

    drift_window = [0] * drift_window_size
    current_drift = 0
    count_since_drift_update = 0
    adjustment = 0

    blink_window = [0] * blink_window_size
    blink_skip_window = 0

    for i in range(0, len(data)):
        drift_window = drift_window[1:]
        drift_window.append(data[i])
        if i != 0 and i % update_interval == 0:
            previous_drift = current_drift
            current_drift = get_slope(drift_window)
            adjustment -= update_interval * previous_drift

            count_since_drift_update = 0
        else:
            count_since_drift_update += 1

        value = data[i] - (count_since_drift_update * current_drift) + adjustment
        filtered.append(value)

        blink_window = blink_window[1:]
        blink_window.append(value)
        if blink_skip_window <= 0:
            is_blink, blink_center = get_blink(blink_window)
            if is_blink:
                blinks.append(i - blink_window_size + blink_center)
                blink_skip_window = blink_window_size - blink_center
        else:
            blink_skip_window -= 1

    return filtered, [[], []], blinks


def get_slope(data):
    if len(data) < 2:
        return 0
    median_size = max(1, len(data) / 20)  # 5%

    start = int(np.median(data[:median_size]))
    end = int(np.median(data[-median_size:]))
    return (end - start) / (len(data) - 1)


def get_continuous_slope(data):
    if len(data) < 2:
        return 0
    slopes = []
    for i in range(1, len(data)):
        slopes.append(data[i] - data[i-1])
    return np.median(slopes)


def get_baseline(data):
    median_size = max(1, len(data) / 20)  # 5%
    start = int(np.median(data[:median_size]))
    end = int(np.median(data[-median_size:]))
    return (end + start) / 2


def get_blink(data):
    max_index = -1
    min_index = -1
    max_value = -sys.maxint - 1
    min_value = sys.maxint
    for i in range(0, len(data)):
        if data[i] > max_value:
            max_value = data[i]
            max_index = i
        if data[i] < min_value:
            min_value = data[i]
            min_index = i

    is_maxima = max_index != 0 and max_index != len(data) -1 \
        and data[max_index - 1] < data[max_index] > data[max_index + 1]
    base = data[:len(data) * 1/3] + data[len(data) * 2/3:]
    is_tall_enough = max_value - np.median(base) > MIN_BLINK_HEIGHT
    # is_tall_enough = max_value - min_value > MIN_BLINK_HEIGHT
    is_centered = (len(data) * 1/3) < max_index < (len(data) * 2/3)

    if is_tall_enough and is_maxima and is_centered:
        return True, max_index
    return False, 0


def plot(figure, row_col, data, color, x=[], max_y=0, min_y=-1, start=0, twin=False, window=SAMPLING_RATE * 60):
    if len(data) == 0:
        return

    chart = figure.add_subplot(row_col)
    if twin:
        chart = chart.twinx()
    chart.set_xticks(np.arange(0, len(data), window / 15))
    if len(x) == 0:
        chart.plot(data, linewidth=1, color=color)
    else:
        chart.scatter(x, data, linewidth=1, color=color)
    chart.set_xbound([start, start + window])
    if max_y and min_y == -1:
        chart.set_ybound([-max_y, max_y])
    elif max_y:
        chart.set_ybound([min_y, max_y])


if __name__ == "__main__":
    main()
