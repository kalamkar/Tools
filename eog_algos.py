'''
Created on Dec 02, 2016

@author: abhi
'''

import csv
import sys
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

    filtered1, markers1 = filter_drift(raw1)
    filtered2, markers2 = filter_drift(raw2)

    slice_start = 10
    slice_end = len(raw1) - 10

    markers1 = markers2 = []
    # markers1 = markers1[slice_start:slice_end]
    # markers2 = markers2[slice_start:slice_end]
    # channel1 = channel2 = []
    channel1 = filtered1[slice_start:slice_end]
    channel2 = filtered2[slice_start:slice_end]

    raw1 = signal.detrend(raw1)
    raw2 = signal.detrend(raw2)
    raw1 = raw1[slice_start:slice_end]
    raw2 = raw2[slice_start:slice_end]

    plot(figure, 211, channel1, 'blue', window=len(channel1))
    plot(figure, 211, markers1, 'red', window=len(markers1))
    plot(figure, 211, raw1, 'lightblue', window=len(raw1), twin=True)

    plot(figure, 212, channel2, 'green', window=len(channel2))
    plot(figure, 212, markers2, 'red', window=len(markers2))
    plot(figure, 212, raw2, 'lightgreen', window=len(raw2), twin=True)


def filter_drift_slopes(data, slope_window_size=5, drift_window_size=500, update_interval=500):
    slopes = []
    filtered = []

    slope_window = data[:slope_window_size - 1].tolist()
    slope_slope_window = [0] * slope_window_size

    drift_window = [0] * drift_window_size
    current_drift = 0
    count_since_drift_update = 0
    adjustment = data[0]

    baselines = []
    baseline = 0

    threshold = 400

    for i in range(0, len(data)):
        drift_window = drift_window[1:]
        drift_window.append(data[i])
        if i != 0 and i % update_interval == 0:
            previous_drift = current_drift
            current_drift = get_slope(drift_window)
            threshold = max(400, abs(current_drift) * 2)
            adjustment -= update_interval * previous_drift

            baseline = get_baseline(drift_window)

            count_since_drift_update = 0
        else:
            count_since_drift_update += 1

        slope_window = slope_window[1:]
        slope_window.append(data[i])
        slope = get_slope(slope_window)

        if abs(slope) > threshold:
            value = data[i] - (count_since_drift_update * current_drift) + adjustment
            filtered.append(value)
            slopes.append(slope)
        else:
            filtered.append(filtered[i - 1] if i > 0 else 0)
            slopes.append(0)

        baselines.append(baseline - ((count_since_drift_update + drift_window_size / 2) * current_drift) + adjustment)

    return filtered, slopes


def filter_drift(data, drift_window_size=500, update_interval=500):
    filtered = []

    drift_window = [0] * drift_window_size
    current_drift = 0
    count_since_drift_update = 0
    adjustment = 0

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

    return filtered, []


def filter_slopes(data, slope_window_size=5):
    slopes = [0] * slope_window_size
    slope_slopes = [0] * slope_window_size
    filtered = [0] * slope_window_size

    thresholds = []
    slope_window = data[:slope_window_size - 1].tolist()
    slope_slope_window = [0] * slope_window_size

    drift_window = [0] * 5000
    threshold = 0
    current_drift = 0
    count_since_drift_update = 0

    for i in range(slope_window_size, len(data)):
        slope_window = slope_window[1:]
        slope_window.append(data[i])
        slope = get_slope(slope_window)

        slope_slope_window = slope_slope_window[1:]
        slope_slope_window.append(slope)
        slope_slope = get_slope(slope_slope_window)

        drift_window = drift_window[1:]
        drift_window.append(data[i])
        if i % 100 == 0:
            threshold = max(400, abs(get_slope(drift_window)) * 2)
        thresholds.append(threshold)

        if abs(slope) > abs(threshold):
            slopes.append(slope)
        else:
            slopes.append(0)

        if abs(slope_slope) > 100:
            slope_slopes.append(slope_slope)
            count_since_drift_update = 0
            current_drift = get_slope(drift_window)
        else:
            slope_slopes.append(0)
            count_since_drift_update += 1

        if abs(slope_slope) > 100:
            value = data[i] - (count_since_drift_update * current_drift)
            filtered.append(value)
        else:
            filtered.append(filtered[i-1])

    return filtered, slope_slopes


def get_slope(data):
    median_size = max(1, len(data) / 20)  # 5%
    start = int(np.median(data[:median_size]))
    end = int(np.median(data[-median_size:]))
    return (end - start) / len(data)


def get_baseline(data):
    median_size = max(1, len(data) / 20)  # 5%
    start = int(np.median(data[:median_size]))
    end = int(np.median(data[-median_size:]))
    return (end + start) / 2


def plot(figure, row_col, data, color, max_y=0, min_y=-1, start=0, twin=False, window=SAMPLING_RATE * 60):
    if len(data) == 0:
        return

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
