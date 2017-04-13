'''
Created on Dec 02, 2016

@author: abhi
'''

import csv
import sys
import math
import numpy as np
import time
import scipy.signal
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
    raw1 = np.array(columns[0][1:]).astype(np.int)
    raw2 = np.array(columns[1][1:]).astype(np.int)

    filtered1 = np.array(columns[2][1:]).astype(np.int)
    filtered2 = np.array(columns[3][1:]).astype(np.int)

    est_col = np.array(columns[4][1:]).astype(np.int)
    est_row = np.array(columns[5][1:]).astype(np.int)

    col = np.array(columns[6][1:]).astype(np.int)
    row = np.array(columns[7][1:]).astype(np.int)

    slice_start = 6000
    slice_end = len(raw1) - 500

    # raw1 = scipy.signal.detrend(raw1)
    # raw2 = scipy.signal.detrend(raw2)

    raw1 = raw1[slice_start:slice_end]
    raw2 = raw2[slice_start:slice_end]

    filtered1 = filtered1[slice_start:slice_end]
    filtered2 = filtered2[slice_start:slice_end]

    row = [4 - x for x in row[slice_start:slice_end]]
    col = [4 - x for x in col[slice_start:slice_end]]

    est_col = est_col[slice_start:slice_end]
    est_row = est_row[slice_start:slice_end]

    print 'Accuracy for Horizontal is %.2f%% and Vertical is %.2f%%' \
          % (get_accuracy(est_col, col), get_accuracy(est_row, row))

    plot(figure, 211, filtered1, 'lightblue', window=len(filtered1))
    # plot(figure, 211, col, 'orange', window=len(col), twin=True)
    # plot(figure, 211, levels1, 'red', window=len(levels1), twin=True)
    plot(figure, 211, raw1, 'blue', window=len(raw1), twin=True)

    plot(figure, 212, filtered2, 'lightgreen', window=len(filtered2))
    # plot(figure, 212, row, 'orange', window=len(row), twin=True)
    # plot(figure, 212, levels2, 'red', window=len(levels2), twin=True)
    plot(figure, 212, raw2, 'green', window=len(raw2), twin=True)


def get_accuracy(estimate, truth, interval=5):
    checks = 0
    successes = 0
    for i in range(len(estimate)):
        if i % interval == 0:
            checks += 1
            successes += 1 if estimate[i] == truth[i] else 0
    return successes * 100.0 / checks


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
