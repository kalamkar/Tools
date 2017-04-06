'''
Created on Dec 02, 2016

@author: abhi
'''

import csv
import sys
import math
import numpy as np
from scipy import signal
from matplotlib import pyplot


SAMPLING_RATE = 51.2
DOWN_SAMPLE_FACTOR = 30
POLY_FIT_WINDOW = 10
MEDIAN_WINDOW = 20
MIN_BLINK_HEIGHT = 10000
START = int(SAMPLING_RATE * 15)  # Ignore first 15 seconds


def main():
    reader = csv.reader(open(sys.argv[1], 'rb'))
    columns = list(zip(*reader))

    figure = pyplot.figure(figsize=(15, 10))

    # show_algo_realtime(columns, figure)
    # show_algo_chunks(columns, figure)
    # show_algo(columns, figure)
    # show_detrend(columns, figure)
    # show_remove_slope(columns, figure)
    show_slope(columns, figure)
    # show_raw(columns, figure)
    # show_filtered(columns, figure)
    # show_filtered_slice(columns, figure)

    pyplot.show()


def show_algo_realtime(columns, figure):
    max_y = 10000
    channel1 = np.array(columns[0][START:]).astype(np.int)
    channel11, channel12 = process_realtime(channel1, 200)
    plot(figure, 211, channel11, 'lightblue', max_y=max_y, start=3000-START)
    plot(figure, 211, channel12, 'blue', max_y=max_y, start=3000-START, twin=True)

    channel2 = np.array(columns[1][START:]).astype(np.int)
    channel21, channel22 = process_realtime(channel2, 200)
    plot(figure, 212, channel21, 'lightgreen', max_y=max_y, start=3000-START)
    plot(figure, 212, channel22, 'green', max_y=max_y, start=3000-START, twin=True)


def show_algo_chunks(columns, figure):
    max_y = 10000
    channel1 = np.array(columns[0][START:]).astype(np.int)
    channel11, channel12 = process_chunks(channel1, 150)
    plot(figure, 211, channel11, 'lightblue', max_y=max_y, start=3000-START)
    plot(figure, 211, channel12, 'blue', max_y=max_y, start=3000-START)

    channel2 = np.array(columns[1][START:]).astype(np.int)
    channel21, channel22 = process_chunks(channel2, 150)
    plot(figure, 212, channel21, 'lightgreen', max_y=max_y, start=3000-START)
    plot(figure, 212, channel22, 'green', max_y=max_y, start=3000-START)


def show_algo(columns, figure):
    max_y = 10000
    channel1 = np.array(columns[0][START:]).astype(np.int)
    channel1 = remove_drift(channel1)
    channel1 = median_filter(channel1)
    plot(figure, 211, channel1, 'lightblue', max_y=max_y, start=3000-START)
    channel1 = diffdiff_filter(channel1, 150)
    plot(figure, 211, channel1, 'blue', max_y=max_y, start=3000-START)

    channel2 = np.array(columns[1][START:]).astype(np.int)
    channel2 = remove_drift(channel2)
    channel2 = median_filter(channel2)
    plot(figure, 212, channel2, 'lightgreen', max_y=max_y, start=3000-START)
    channel2 = diffdiff_filter(channel2, 150)
    plot(figure, 212, channel2, 'green', max_y=max_y, start=3000-START)


def show_filtered(columns, figure):
    channel1 = columns[2][START:-50]
    channel2 = columns[3][START:-50]

    plot(figure, 211, channel1, 'blue', start=3000-START)
    plot(figure, 212, channel2, 'green', start=3000-START)


def show_filtered_slice(columns, figure):
    channel1 = columns[2][7000:9000]
    channel2 = columns[3][7000:9000]

    plot(figure, 211, channel1, 'blue', window=len(channel1))
    plot(figure, 212, channel2, 'green', window=len(channel2))


def show_remove_slope(columns, figure):
    raw1 = np.array(columns[0][START:-50]).astype(np.int)
    raw2 = np.array(columns[1][START:-50]).astype(np.int)

    channel1 = remove_slope(raw1)[7000:9000]
    channel2 = remove_slope(raw2)[7000:9000]

    raw1 = signal.detrend(raw1)[7000:9000]
    raw2 = signal.detrend(raw2)[7000:9000]

    plot(figure, 211, raw1, 'lightblue', window=len(raw1))
    plot(figure, 211, channel1, 'blue', window=len(channel1), twin=True)
    plot(figure, 212, raw2, 'lightgreen', window=len(raw2))
    plot(figure, 212, channel2, 'green', window=len(channel2), twin=True)


def show_slope(columns, figure):
    raw1 = np.array(columns[0][START:-50]).astype(np.int)
    raw2 = np.array(columns[1][START:-50]).astype(np.int)

    slice_start = 7000
    slice_end = 9000

    slope1 = get_slopes(raw1)[slice_start:slice_end]
    slope2 = get_slopes(raw2)[slice_start:slice_end]

    raw1 = signal.detrend(raw1)[slice_start:slice_end]
    raw2 = signal.detrend(raw2)[slice_start:slice_end]
    # raw1 = diffdiff(raw1, cutoff=2000)[slice_start:slice_end]
    # raw2 = diffdiff(raw2, cutoff=2000)[slice_start:slice_end]

    plot(figure, 211, raw1, 'lightblue', window=len(raw1))
    plot(figure, 211, slope1, 'blue', window=len(slope1), twin=True)
    plot(figure, 212, raw2, 'lightgreen', window=len(raw2))
    plot(figure, 212, slope2, 'green', window=len(slope2), twin=True)


def show_raw(columns, figure):
    raw1 = np.array(columns[0][START:-50]).astype(np.int)
    raw2 = np.array(columns[1][START:-50]).astype(np.int)

    raw1 = signal.detrend(raw1)
    raw2 = signal.detrend(raw2)

    plot(figure, 211, raw1, 'blue', window=len(raw1))
    plot(figure, 212, raw2, 'green', window=len(raw2))


def show_detrend(columns, figure):
    channel1 = np.array(columns[0][START:-50]).astype(np.int)
    channel2 = np.array(columns[1][START:-50]).astype(np.int)

    channel1 = signal.detrend(channel1, bp=range(0, len(channel1), POLY_FIT_WINDOW))
    channel2 = signal.detrend(channel2, bp=range(0, len(channel1), POLY_FIT_WINDOW))

    plot(figure, 211, channel1, 'blue', start=3000-START)
    plot(figure, 212, channel2, 'green', start=3000-START)


def diffdiff(data, cutoff=0):
    lastdiff1 = data[1] - data[0]
    filtered = [0, lastdiff1]
    lastdiff2 = 0
    for i in range(2, len(data)):
        diff1 = data[i] - data[i-1]
        diff2 = diff1 - lastdiff1
        filtered.append(diff2 if not cutoff or abs(diff2) > cutoff else 0)
        lastdiff1 = diff1
        lastdiff2 = diff2 if not cutoff or abs(diff2) > cutoff else 0
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


def median_filter(data):
    filtered = [0]
    for i in range(MEDIAN_WINDOW, len(data)):
        filtered.append(int(np.median(data[i - MEDIAN_WINDOW:i])))
    return filtered


def remove_drift(data, degree=2):
    filtered = []
    for i in range(0, len(data) - POLY_FIT_WINDOW, POLY_FIT_WINDOW):
        curve = get_curve(data[i:i + POLY_FIT_WINDOW])
        chunk = []
        for j in range(i, i + POLY_FIT_WINDOW):
            chunk.append(data[j] - np.polyval(curve, i + j))
        chunk = signal.detrend(chunk)
        filtered.extend(chunk)

    return filtered


def get_curve(data, degree=2):
    x = []
    y = []
    for i in range(0, len(data)):
        if i % DOWN_SAMPLE_FACTOR == 0:
            x.append(i)
            y.append(data[i])
    return np.polyfit(x, y, degree)


def remove_slope(data):
    filtered = []
    slope = get_slope(data[:POLY_FIT_WINDOW])
    for i in range(POLY_FIT_WINDOW, len(data)):
        slope = get_slope(data[i - POLY_FIT_WINDOW:i])
        filtered.append(data[i] - (i * slope))
    return filtered


def get_slopes(data):
    slopes = [0] * POLY_FIT_WINDOW
    for i in range(POLY_FIT_WINDOW, len(data)):
        slope = get_slope(data[i - POLY_FIT_WINDOW:i])
        slopes.append(slope)
    return slopes


def get_slopes_with_memories(data):
    slopes = [0] * POLY_FIT_WINDOW
    memory = [0] * POLY_FIT_WINDOW
    for i in range(POLY_FIT_WINDOW, len(data)):
        slope = get_slope(data[i - POLY_FIT_WINDOW:i])
        slopes.append(slope + memory[0])
        memory = memory[1:]
        memory.append(slope)
    return slopes


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


def filter_drift_slopes_flats(data, slope_window_size=5, threshold_multiplier=200, drift_window_size=500):
    slopes = []
    filtered = []

    slope_window = data[:slope_window_size].tolist()

    window_for_thresholds = [0] * drift_window_size
    threshold = 400

    current_drift = 0
    adjustment = 0

    featureless = []

    for i in range(0, len(data)):
        slope_window = slope_window[1:]
        slope_window.append(data[i])
        slope = get_slope(slope_window)

        window_for_thresholds = window_for_thresholds[1:]
        window_for_thresholds.append(data[i])

        if i % drift_window_size == 0:
            long_slope = get_slope(window_for_thresholds)
            try:
                threshold = math.log10(abs(long_slope)) * threshold_multiplier
            except ValueError:
                pass

        if abs(slope) > abs(threshold):
            if len(featureless) > 100:
                previous_drift = current_drift
                current_drift = get_continuous_slope(featureless)
                old_value = data[i] - (i * previous_drift) + adjustment
                new_value = data[i] - (i * current_drift) + adjustment
                adjustment -= new_value - old_value

            filtered.append(data[i] - (i * current_drift) + adjustment)
            slopes.append(slope)

            featureless = []
        else:
            filtered.append(filtered[i - 1] if i > 0 else 0)
            slopes.append(0)

            featureless.append(data[i])

    return filtered, slopes


def filter_drift_slopes_fixed(data, slope_window_size=5, drift_window_size=500,
                              update_interval=500, threshold_multiplier=200):
    slopes = []
    filtered = []

    slope_window = data[:slope_window_size].tolist()

    drift_window = [0] * drift_window_size
    current_drift = 0
    adjustment = 0
    threshold = 400

    mins = []
    maxs = []
    count_since_update = 0
    calibration_window = [0] * drift_window_size
    prev_min = data[0]
    prev_max = data[0]

    for i in range(0, len(data)):
        drift_window = drift_window[1:]
        drift_window.append(data[i])
        if i != 0 and i % update_interval == 0:
            previous_drift = current_drift
            current_drift = get_slope(drift_window)

            threshold = math.log10(abs(current_drift)) * threshold_multiplier

            old_value = data[i] - (i * previous_drift) + adjustment
            new_value = data[i] - (i * current_drift) + adjustment
            adjustment -= new_value - old_value

        value = data[i] - (i * current_drift) + adjustment
        calibration_window = calibration_window[1:]
        calibration_window.append(value)

        if i != 0 and i % update_interval == 0:
            prev_min = int(min(calibration_window))
            prev_max = int(max(calibration_window))

            count_since_update = 0
        else:
            count_since_update += 1

        maxs.append(prev_max)  # - (count_since_update * current_drift) + adjustment)
        mins.append(prev_min)  # - (count_since_update * current_drift) + adjustment)

        slope_window = slope_window[1:]
        slope_window.append(data[i])
        slope = get_slope(slope_window)

        if abs(slope) > abs(threshold):
            filtered.append(value)
            slopes.append(slope)
        else:
            filtered.append(filtered[i - 1] if i > 0 else 0)
            slopes.append(0)

    return filtered, [maxs, mins]


def filter_slopes(data, slope_window_size=5, threshold_multiplier=300, drift_window_size=500):
    slopes = [0] * slope_window_size
    slope_slopes = [0] * slope_window_size
    filtered = [0] * slope_window_size

    thresholds = []
    slope_window = data[:slope_window_size].tolist()
    slope_slope_window = [0] * slope_window_size

    window_for_thresholds = [0] * drift_window_size
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

        window_for_thresholds = window_for_thresholds[1:]
        window_for_thresholds.append(data[i])

        if i % drift_window_size == 0:
            long_slope = get_slope(window_for_thresholds)
            threshold = math.log10(abs(long_slope)) * threshold_multiplier
        thresholds.append(threshold)

        if abs(slope) > abs(threshold):
            slopes.append(slope)
        else:
            slopes.append(0)

        if abs(slope_slope) > threshold:
            slope_slopes.append(slope_slope)
            count_since_drift_update = 0
            current_drift = get_slope(window_for_thresholds)
        else:
            slope_slopes.append(0)
            count_since_drift_update += 1

        if abs(slope_slope) > 100:
            value = data[i] - (count_since_drift_update * current_drift)
            filtered.append(value)
        else:
            filtered.append(filtered[i-1])

    return filtered, slopes


def process_chunks(data, cutoff):
    stage1 = []
    for i in range(0, len(data) - POLY_FIT_WINDOW, POLY_FIT_WINDOW):
        curve = get_curve(data[i:i + POLY_FIT_WINDOW], degree=1)
        chunk = []
        for j in range(i, i + POLY_FIT_WINDOW):
            chunk.append(data[j] - np.polyval(curve, i + j))
        chunk = signal.detrend(chunk)
        stage1.extend(chunk)

    stage2 = []
    for i in range(MEDIAN_WINDOW, len(stage1)):
        stage2.append(int(np.median(stage1[i - MEDIAN_WINDOW:i])))

    lastdiff1 = stage2[1] - stage2[0]
    stage3 = [0, lastdiff1]
    lastdata = stage2[0]
    for i in range(2, len(stage2)):
        diff1 = stage2[i] - stage2[i - 1]
        diff2 = diff1 - lastdiff1
        if abs(diff2) > cutoff:
            lastdata = stage2[i]
        stage3.append(lastdata)
        lastdiff1 = diff1

    return stage2, stage3


def process_realtime(data, cutoff):
    stage1 = []
    stage2 = []
    raw = [0] * POLY_FIT_WINDOW
    straight = [0] * MEDIAN_WINDOW
    last_median = 0
    lastdiff1 = 0
    lastdata = 0
    for i in range(0, len(data)):
        raw = raw[1:]
        raw.append(data[i])
        curve = get_curve(raw, degree=1)

        straight = straight[1:]
        straight.append(raw[len(raw) - 1] - np.polyval(curve, len(raw)))

        median = int(np.median(straight))
        stage1.append(median)

        diff1 = median - last_median
        diff2 = diff1 - lastdiff1
        if abs(diff2) > cutoff:
            lastdata = median
        stage2.append(lastdata)

        last_median = median
        lastdiff1 = diff1

    return stage1, stage2


def filter_drift_with_blinks(data, drift_window_size=500, update_interval=500, blink_window_size=30):
    filtered = []
    blinks = []

    drift_window = [0] * drift_window_size
    current_drift = 0
    count_since_drift_update = 0
    adjustment = 0

    blink_window = [0] * blink_window_size
    blink_skip_window = 0

    baseline = []
    count_since_calibration_update = 0
    calibration_window = [0] * drift_window_size
    prev_base = data[0]

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

        calibration_window = calibration_window[1:]
        calibration_window.append(value)

        if i != 0 and i % 100 == 0:
            prev_base = int(np.median(calibration_window))
            count_since_calibration_update = 0
        else:
            count_since_calibration_update += 1

        baseline.append(prev_base) # - (count_since_calibration_update * current_drift))

        blink_window = blink_window[1:]
        blink_window.append(value)
        if blink_skip_window <= 0:
            is_blink, blink_center = get_blink(blink_window)
            if is_blink:
                blinks.append(i - blink_window_size + blink_center)
                blink_skip_window = blink_window_size - blink_center
        else:
            blink_skip_window -= 1

    return filtered, baseline, [], blinks


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
