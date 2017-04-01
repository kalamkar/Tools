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
DOWN_SAMPLE_FACTOR = 30
POLY_FIT_WINDOW = 50
MEDIAN_WINDOW = 20

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

    slope1 = get_slopes(raw1)[9000:9500]
    slope2 = get_slopes(raw2)[9000:9500]

    raw1 = signal.detrend(raw1)[9000:9500]
    raw2 = signal.detrend(raw2)[9000:9500]

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


def get_slope(data):
    median_size = max(1, POLY_FIT_WINDOW / 20) # 5%
    start = int(np.median(data[:median_size]))
    end = int(np.median(data[-median_size:]))
    return float((end - start) / len(data))


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
