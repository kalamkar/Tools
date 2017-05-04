'''
Created on Dec 02, 2016

@author: abhi
'''

import csv
import sys
import math
import numpy as np
import time
import scipy.stats
import scipy.signal
from matplotlib import pyplot
from common import plot

SAMPLING_RATE = 51.2
MIN_BLINK_HEIGHT = 10000
GESTURE_VISIBILITY_MILLIS = 2000


def main():
    reader = csv.reader(open(sys.argv[1], 'rb'))
    columns = list(zip(*reader))

    figure = pyplot.figure(figsize=(15, 10))

    start = 1
    end = -1
    if len(sys.argv) > 3:
        start = int(sys.argv[2])
        end = int(sys.argv[3])

    start_time = time.time()
    show(columns, figure, start, end)
    print 'Processing took %d seconds' % (time.time() - start_time)

    pyplot.show()


def show(columns, figure, start, end):
    raw1 = np.array(columns[0][1:]).astype(np.int)
    raw2 = np.array(columns[1][1:]).astype(np.int)

    [filtered1, markers11, markers12], \
    [filtered2, markers21, markers22] = process(raw1, raw2)

    filtered1 = filtered1[start:end]
    filtered2 = filtered2[start:end]

    raw1 = scipy.signal.detrend(raw1)
    raw2 = scipy.signal.detrend(raw2)

    raw1 = raw1[start:end]
    raw2 = raw2[start:end]

    markers11 = markers11[start:end]
    markers21 = markers21[start:end]

    markers12 = markers12[start:end]
    markers22 = markers22[start:end]

    plot(figure, 211, markers11, 'yellow', window=len(markers11), twin=True)
    plot(figure, 211, markers12, 'orange', window=len(markers12), twin=True)
    # plot(figure, 211, raw1, 'lightblue', window=len(raw1))
    plot(figure, 211, filtered1, 'blue', window=len(filtered1))

    plot(figure, 212, markers21, 'yellow', window=len(markers21), twin=True)
    plot(figure, 212, markers22, 'orange', window=len(markers22), twin=True)
    # plot(figure, 212, raw2, 'lightgreen', window=len(raw2))
    plot(figure, 212, filtered2, 'green', window=len(filtered2))


def process(horizontal, vertical):
    h_filtered = []
    v_filtered = []

    h_drift1 = FixedWindowSlopeRemover(window_size=512)
    v_drift1 = FixedWindowSlopeRemover(window_size=512)

    # h_gestures = DiffDiffGestureFilter()
    # v_gestures = DiffDiffGestureFilter()

    h_gestures = SlopeGestureFilter()
    v_gestures = SlopeGestureFilter()

    for i in range(0, len(horizontal)):

        h_value = horizontal[i]
        v_value = vertical[i]

        h_value = h_drift1.update(h_value)
        v_value = v_drift1.update(v_value)

        h_value = h_gestures.update(h_value)
        v_value = v_gestures.update(v_value)

        h_filtered.append(h_value)
        v_filtered.append(v_value)

    return [h_filtered, h_gestures.features, []], \
           [v_filtered, v_gestures.features, []]


class SlopeGestureFilter:
    def __init__(self, slope_window_size=5, threshold_multiplier=3, threshold_update_interval=512, min_threshold=1000):
        self.slope_window_size = slope_window_size
        self.slope_window = [0] * slope_window_size

        self.threshold = 0

        self.threshold_update_interval = threshold_update_interval
        self.threshold_multiplier = threshold_multiplier
        self.threshold_window = []
        self.update_count = 0

        self.min_threshold = min_threshold

        self.skip_window = 0

        self.features = []

    def update(self, value):
        self.update_count += 1

        self.slope_window = self.slope_window[1:]
        self.slope_window.append(value)
        slope = get_slope(self.slope_window)

        self.threshold_window.append(slope)
        if self.update_count % self.threshold_update_interval == 0:
            self.threshold = np.std(self.threshold_window) * self.threshold_multiplier
            self.threshold = max(self.min_threshold, self.threshold)
            self.threshold_window = []

        if self.skip_window <= 0 and abs(slope) > abs(self.threshold):
            self.features.append(slope)
            self.skip_window = SAMPLING_RATE * (GESTURE_VISIBILITY_MILLIS / 1000)
        else:
            self.features.append(0)
            self.skip_window -= 1 if self.skip_window > 0 else 0

        return value


class DiffDiffGestureFilter:
    def __init__(self):
        self.prev_value = 0
        self.prev_diff = 0

        self.features = []

    def update(self, value):
        diff = value - self.prev_value
        diff_diff = diff - self.prev_diff

        self.prev_value = value
        self.prev_diff = diff

        self.features.append(diff_diff)
        return value


class FixedWindowSlopeRemover:
    def __init__(self, window_size=500):
        self.window_size = window_size
        self.window = []
        self.current_drift = 0
        self.adjustment = 0

    def update(self, raw):
        self.window.append(raw)

        if len(self.window) == self.window_size:
            previous_drift = self.current_drift
            self.current_drift = get_slope(self.window)
            self.adjustment -= self.window_size * previous_drift
            self.window = []

        return raw - (len(self.window) * self.current_drift) + self.adjustment

    def remove_spike(self, size):
        remove_spike(self.window, size)


def get_slope(data):
    if len(data) < 2:
        return 0
    median_size = max(1, len(data) / 20)  # 5%

    start = int(np.median(data[:median_size]))
    end = int(np.median(data[-median_size:]))
    return float((end - start) / (len(data) - 1))


def get_min_max(data):
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

    return min_value, min_index, max_value, max_index


def get_blink(data):
    min_value, min_index, max_value, max_index = get_min_max(data)

    is_maxima = max_index != 0 and max_index != len(data) - 1 \
        and data[max_index - 1] < data[max_index] > data[max_index + 1]
    base = data[:len(data) * 1/3] + data[len(data) * 2/3:]
    is_tall_enough = max_value - np.median(base) > MIN_BLINK_HEIGHT
    # is_tall_enough = max_value - min_value > MIN_BLINK_HEIGHT
    is_centered = (len(data) * 1/3) < max_index < (len(data) * 2/3)

    if is_tall_enough and is_maxima and is_centered:
        return True, max_index
    return False, 0


def remove_spike(data, size):
    end = len(data) - 1
    if end <= 0:
        return
    start = max(0, end - size)
    slope = (data[end] - data[start]) / size
    for i in range(start, end + 1):
        data[i] = data[start] + int(slope * (i - start))


if __name__ == "__main__":
    main()
