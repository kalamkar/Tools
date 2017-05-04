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
from common import *


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
    show(columns, figure, start, end, start_time)

    pyplot.show()


def show(columns, figure, start, end, start_time):
    raw1 = np.array(columns[0][1:]).astype(np.int)
    raw2 = np.array(columns[1][1:]).astype(np.int)

    [filtered1, markers11, markers12], \
    [filtered2, markers21, markers22] = process(raw1, raw2)
    print 'Processing took %d seconds' % (time.time() - start_time)

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

    # plot(figure, 211, raw1, 'lightblue', window=len(raw1), twin=True)
    plot(figure, 211, filtered1, 'blue', window=len(filtered1))
    plot(figure, 211, markers11, 'yellow', window=len(markers11), twin=True, max_y=1000)
    plot(figure, 211, markers12, 'orange', window=len(markers12), twin=True, bar=True, max_y=1000)

    # plot(figure, 212, raw2, 'lightgreen', window=len(raw2), twin=True)
    plot(figure, 212, filtered2, 'green', window=len(filtered2))
    plot(figure, 212, markers21, 'yellow', window=len(markers21), twin=True, max_y=1000)
    plot(figure, 212, markers22, 'orange', window=len(markers22), twin=True, bar=True, max_y=1000)


def process(horizontal, vertical):
    h_filtered = []
    v_filtered = []

    h_drift1 = MedianMovingSlopeDriftRemover(threshold_window_size=256)
    v_drift1 = MedianMovingSlopeDriftRemover(threshold_window_size=256)
    # h_drift1 = FixedWindowSlopeRemover(window_size=512)
    # v_drift1 = FixedWindowSlopeRemover(window_size=512)

    h_saccades = SlopeGestureFilter(window_size=5)
    v_saccades = SlopeGestureFilter(window_size=5)

    for i in range(0, len(horizontal)):

        h_value = horizontal[i]
        v_value = vertical[i]

        h_value = h_drift1.update(h_value)
        v_value = v_drift1.update(v_value)

        # h_value = h_saccades.update(h_value)
        # v_value = v_saccades.update(v_value)

        h_filtered.append(h_value)
        v_filtered.append(v_value)

    return [h_filtered, h_saccades.thresholds, h_saccades.features], \
           [v_filtered, v_saccades.thresholds, v_saccades.features]


class SlopeGestureFilter:
    def __init__(self, window_size=5, threshold_window_size=512, threshold_multiplier=1.0):
        self.window = [0] * window_size

        self.threshold = [0, 0]
        self.threshold_multiplier = threshold_multiplier
        self.threshold_window = [0] * threshold_window_size
        self.count_since_threshold_update = 0

        self.thresholds = []
        self.features = []

    def update(self, value):
        self.window = self.window[1:]
        self.window.append(value)
        diff = get_slope(self.window)

        self.threshold_window = self.threshold_window[1:]
        self.threshold_window.append(diff)
        self.count_since_threshold_update += 1
        if self.count_since_threshold_update == len(self.threshold_window):
            median = np.median(self.threshold_window)
            stddev = np.std(self.threshold_window)
            low = 0 if median > 0 else median - stddev
            high = 0 if median < 0 else median + stddev
            self.threshold = [low, high]
            self.count_since_threshold_update = 0

        self.thresholds.append(self.threshold)
        self.features.append(diff if diff < self.threshold[0] or diff > self.threshold[1] else 0)
        return value


class DiffSlopeGestureFilter:
    def __init__(self, window_size=5, threshold_window_size=512, threshold_multiplier=1.0):
        self.window = [0] * window_size
        self.prev_diff = 0

        self.threshold = 0
        # self.threshold_multiplier = threshold_multiplier
        # self.threshold_window = [0] * threshold_window_size
        # self.count_since_threshold_update = 0

        self.thresholds = []
        self.features = []

    def update(self, value):
        self.window = self.window[1:]
        self.window.append(value)
        diff = get_slope(self.window)
        diff_diff = diff - self.prev_diff

        self.prev_diff = diff

        # self.threshold_window = self.threshold_window[1:]
        # self.threshold_window.append(diff_diff)
        # self.count_since_threshold_update += 1
        # if self.count_since_threshold_update == len(self.threshold_window):
        #     self.count_since_threshold_update = 0
        #     self.threshold = self.threshold_multiplier * np.std(self.threshold_window)

        self.thresholds.append(self.threshold)
        self.features.append(diff_diff if abs(diff_diff) > self.threshold else 0)
        return value


class FixedWindowSlopeRemover:
    def __init__(self, window_size=512):
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


class MedianMovingSlopeDriftRemover:
    def __init__(self, slope_window_size=5, threshold_window_size=512):
        self.threshold_window_size = threshold_window_size
        self.threshold_window = []
        self.slope_window = [0] * slope_window_size
        self.current_drift = 0
        self.adjustment = 0

    def update(self, raw):
        self.slope_window = self.slope_window[1:]
        self.slope_window.append(raw)
        slope = get_slope(self.slope_window)

        self.threshold_window.append(slope)
        if len(self.threshold_window) == self.threshold_window_size:
            previous_drift = self.current_drift
            # self.current_drift = get_slope(self.threshold_window)
            self.current_drift = np.median(self.threshold_window)
            self.adjustment -= self.threshold_window_size * previous_drift
            self.threshold_window = []

        return raw - (len(self.threshold_window) * self.current_drift) + self.adjustment


if __name__ == "__main__":
    main()
