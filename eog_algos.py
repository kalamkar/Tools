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
from matplotlib import pyplot

SAMPLING_RATE = 51.2
MIN_BLINK_HEIGHT = 10000


def main():
    reader = csv.reader(open(sys.argv[1], 'rb'))
    columns = list(zip(*reader))

    figure = pyplot.figure(figsize=(15, 10))

    start = time.time()
    show(columns, figure)
    print 'Processing took %d seconds' % (time.time() - start)

    pyplot.show()


def show(columns, figure):
    col = np.array(columns[6][1:]).astype(np.int)
    row = np.array(columns[7][1:]).astype(np.int)

    raw1 = np.array(columns[0][1:]).astype(np.int)
    raw2 = np.array(columns[1][1:]).astype(np.int)

    [filtered1, levels1, markers11, markers12, blink_points1], \
    [filtered2, levels2, markers21, markers22, blink_points2] = process(raw1, raw2)

    slice_start = 6000
    slice_end = len(raw1) - 50

    blink_points1 = [i - slice_start if slice_start <= i < slice_end else 0 for i in blink_points2] # Hack
    blink_points2 = [i - slice_start if slice_start <= i < slice_end else 0 for i in blink_points2]
    blink_values1 = [filtered1[i + slice_start] for i in blink_points1]
    blink_values2 = [filtered2[i + slice_start] for i in blink_points2]

    markers11 = markers11[slice_start:slice_end]
    markers21 = markers21[slice_start:slice_end]

    markers12 = markers12[slice_start:slice_end]
    markers22 = markers22[slice_start:slice_end]

    channel1 = filtered1[slice_start:slice_end]
    channel2 = filtered2[slice_start:slice_end]

    row = [4 - x for x in row[slice_start:slice_end]]
    col = [4 - x for x in col[slice_start:slice_end]]

    levels1 = levels1[slice_start:slice_end]
    levels2 = levels2[slice_start:slice_end]

    print 'Accuracy for Horizontal is %.2f%% and Vertical is %.2f%%' \
          % (get_accuracy(levels1, col), get_accuracy(levels2, row))

    # plot(figure, 211, channel1, 'blue', window=len(channel1))
    # plot(figure, 211, markers11, 'orange', window=len(markers11))
    # plot(figure, 211, markers12, 'yellow', window=len(markers12))
    plot(figure, 211, blink_values1, 'red', x=blink_points1, window=len(channel1))
    plot(figure, 211, col, 'lightblue', window=len(col), twin=True)
    plot(figure, 211, levels1, 'red', window=len(levels1), twin=True)

    # plot(figure, 212, channel2, 'green', window=len(channel2))
    # plot(figure, 212, markers21, 'orange', window=len(markers21))
    # plot(figure, 212, markers22, 'yellow', window=len(markers22))
    plot(figure, 212, blink_values2, 'red', x=blink_points2, window=len(channel2))
    plot(figure, 212, row, 'lightgreen', window=len(row), twin=True)
    plot(figure, 212, levels2, 'red', window=len(levels2), twin=True)


def get_accuracy(estimate, truth, interval=5):
    checks = 0
    successes = 0
    for i in range(len(estimate)):
        if i % interval == 0:
            checks += 1
            successes += 1 if estimate[i] == truth[i] else 0
    return successes * 100.0 / checks


def process(horizontal, vertical, remove_blinks=True, remove_baseline=True):
    h_filtered = []
    v_filtered = []

    h_drift = FixedWindowSlopeRemover()
    v_drift = FixedWindowSlopeRemover()

    h_drift2 = WeightedWindowDriftRemover()
    v_drift2 = WeightedWindowDriftRemover()

    h_feature_tracker = SlopeFeatureTracker()
    v_feature_tracker = SlopeFeatureTracker()

    blink_detector = BlinkDetector()

    h_minmax = FeatureBasedMinMaxTracker()
    v_minmax = FeatureBasedMinMaxTracker()

    for i in range(0, len(horizontal)):
        h_raw = horizontal[i]
        v_raw = vertical[i]

        h_value = h_drift.update(h_raw)
        v_value = v_drift.update(v_raw)

        # h_raw = h_value
        # v_raw = v_value
        # h_value = h_drift2.update(h_value)
        # v_value = v_drift2.update(v_value)

        h_feature_tracker.update(h_raw, h_value)
        v_feature_tracker.update(v_raw, v_value)

        if blink_detector.check(v_value) and remove_blinks:
            h_drift.remove_spike(blink_detector.blink_window_size)
            v_drift.remove_spike(blink_detector.blink_window_size)
            remove_spike(h_filtered, blink_detector.blink_window_size)
            remove_spike(v_filtered, blink_detector.blink_window_size)
            h_minmax.remove_spike(blink_detector.blink_window_size)
            v_minmax.remove_spike(blink_detector.blink_window_size)

        if remove_baseline:
            h_value = h_value - h_feature_tracker.baseline[i]
            v_value = v_value - v_feature_tracker.baseline[i]

        h_minmax.update(h_value)
        v_minmax.update(v_value)

        h_filtered.append(h_value)
        v_filtered.append(v_value)

    return [h_filtered, h_minmax.levels, h_minmax.mins, h_minmax.maxs, blink_detector.blink_indices], \
           [v_filtered, v_minmax.levels, v_minmax.mins, v_minmax.maxs, blink_detector.blink_indices]


class FixedWindowSlopeRemover:
    def __init__(self, drift_window_size=500):
        self.drift_window_size = drift_window_size
        self.drift_window = []
        self.current_drift = 0
        self.adjustment = 0

    def update(self, raw):
        self.drift_window.append(raw)

        if len(self.drift_window) == self.drift_window_size:
            previous_drift = self.current_drift
            self.current_drift = get_slope(self.drift_window)
            self.adjustment -= self.drift_window_size * previous_drift
            self.drift_window = []

        return raw - (len(self.drift_window) * self.current_drift) + self.adjustment

    def remove_spike(self, size):
        remove_spike(self.drift_window, size)


class SlopeFeatureTracker:
    def __init__(self, slope_window_size=5, threshold_multiplier=300, threshold_update_interval=500):
        self.slope_window_size = slope_window_size
        self.threshold_update_interval = threshold_update_interval
        self.slope_window = [0] * slope_window_size
        self.threshold = threshold_multiplier
        self.threshold_multiplier = threshold_multiplier
        self.features = []
        self.baseline = []
        self.thresholds = []
        self.drift_window = []

        self.feature_adjustment = 0
        self.update_count = 0

    def update(self, raw, flattened):
        self.update_count += 1

        self.slope_window = self.slope_window[1:]
        self.slope_window.append(raw)
        slope = get_slope(self.slope_window)

        self.drift_window.append(raw)
        if self.update_count % self.threshold_update_interval == 0:
            drift = get_slope(self.drift_window)
            if abs(drift) > 0:
                self.threshold = math.log10(abs(drift)) * self.threshold_multiplier
            self.drift_window = []

        self.thresholds.append(self.threshold)

        if abs(slope) > abs(self.threshold):
            self.features.append(slope)
            if self.update_count > self.slope_window_size:
                self.feature_adjustment = flattened - self.baseline[self.update_count - self.slope_window_size]
            else:
                self.feature_adjustment = 0
        else:
            self.features.append(0)

        self.baseline.append(flattened - self.feature_adjustment)


class FixedWindowMinMaxTracker:
    def __init__(self, window_size=500, num_steps=5):
        self.num_steps = num_steps
        self.window_size = window_size
        self.minmax_window = []
        self.current_drift = 0

        self.current_min = 0
        self.current_max = 0

        self.maxs = []
        self.mins = []

        self.levels = []

    def update(self, value):
        self.minmax_window.append(value)

        if len(self.minmax_window) == self.window_size:
            self.current_drift = get_slope(self.minmax_window)
            stddev = np.std(self.minmax_window)

            min_value, min_index, max_value, max_index = get_min_max(self.minmax_window)
            min_value += (len(self.minmax_window) - min_index) * self.current_drift
            max_value += (len(self.minmax_window) - max_index) * self.current_drift
            self.current_min = (min_value + max_value) / 2 - 8000
            self.current_max = (min_value + max_value) / 2 + 8000
            self.minmax_window = []

        self.mins.append(self.current_min)
        self.maxs.append(self.current_max)

        self.levels.append(get_level(value, self.current_min, self.current_max, self.num_steps))

    def remove_spike(self, size):
        remove_spike(self.minmax_window, size)


class FeatureBasedMinMaxTracker:
    def __init__(self, window_size=500, num_steps=5):
        self.num_steps = num_steps
        self.window_size = window_size
        self.window = []
        self.current_drift = 0

        self.current_min = 0
        self.current_max = 0

        self.maxs = []
        self.mins = []

        self.levels = []

    def update(self, value):
        self.window.append(value)

        if len(self.window) >= self.window_size and value != self.window[-2]:
            self.current_drift = get_slope(self.window)
            # stddev = np.std(self.window)

            min_value, min_index, max_value, max_index = get_min_max(self.window)
            min_value += (len(self.window) - min_index) * self.current_drift
            max_value += (len(self.window) - max_index) * self.current_drift
            self.current_min = (min_value + max_value) / 2 - 8000
            self.current_max = (min_value + max_value) / 2 + 8000
            self.window = []

        self.mins.append(self.current_min)
        self.maxs.append(self.current_max)

        self.levels.append(get_level(value, self.current_min, self.current_max, self.num_steps))

    def remove_spike(self, size):
        remove_spike(self.window, size)


class BlinkDetector:
    def __init__(self, blink_window_size=50):
        self.blink_window_size = blink_window_size
        self.blink_indices = []

        self.blink_window = [0] * blink_window_size
        self.blink_skip_window = 0

        self.update_count = 0

    def check(self, flattened):
        self.update_count += 1

        self.blink_window = self.blink_window[1:]
        self.blink_window.append(flattened)

        is_blink = False
        if self.blink_skip_window <= 0:
            is_blink, blink_center = get_blink(self.blink_window)
            if is_blink:
                # self.blink_indices.append(self.update_count - self.blink_window_size)
                self.blink_indices.append(self.update_count - self.blink_window_size + blink_center)
                # self.blink_indices.append(self.update_count)
                self.blink_skip_window = self.blink_window_size - blink_center
        else:
            self.blink_skip_window -= 1

        return is_blink


class WeightedWindowDriftRemover:
    def __init__(self, window_size=1024):
        self.window_size = window_size
        self.window = [1] * window_size
        self.update_count = 0
        self.baseline = []

        xx = np.linspace(-3, 0, num=self.window_size)
        self.window_mask = scipy.stats.norm.pdf(xx, loc=0, scale=1)
        self.window_mask = self.window_size * self.window_mask / sum(self.window_mask)
        # self.window_mask = ones(1, self.window_size)

    def update(self, value):
        self.update_count += 1

        self.window = self.window[1:]
        self.window.append(value)

        mean_estimate = np.mean(self.window_mask * self.window)

        self.baseline.append(mean_estimate)
        return value - mean_estimate

        # avgWin = zeros(1, winlen);
        # if kk > winlen
        #     avgWin = z(kk - winlen:kk - 1)
        # else:
        #     avgWin(winlen - kk + 1:end) = z(1:kk)
        #
        # meanEst(kk) = mean(winmask.* avgWin)
        #
        # modz(kk) = z(kk) - meanEst(kk)


def get_slope(data):
    if len(data) < 2:
        return 0
    median_size = max(1, len(data) / 20)  # 5%

    start = int(np.median(data[:median_size]))
    end = int(np.median(data[-median_size:]))
    return float((end - start) / (len(data) - 1))


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


def get_level(value, min_value, max_value, num_steps):
    size = (max_value - min_value) / num_steps
    value = max(min_value, min(max_value, value))
    return (value - min_value) / size if size > 0 else 0


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

    chart.margins(0.0, 0.05)
    chart.grid(not twin)


if __name__ == "__main__":
    main()
