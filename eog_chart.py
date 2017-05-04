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
from common import plot

SAMPLING_RATE = 51.2
MIN_BLINK_HEIGHT = 10000


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

    filtered1 = np.array(columns[2][1:]).astype(np.int)
    filtered2 = np.array(columns[3][1:]).astype(np.int)

    # est_col = np.array(columns[4][1:]).astype(np.int)
    # est_row = np.array(columns[5][1:]).astype(np.int)
    #
    # col = np.array(columns[6][1:]).astype(np.int)
    # row = np.array(columns[7][1:]).astype(np.int)

    raw1 = scipy.signal.detrend(raw1)
    raw2 = scipy.signal.detrend(raw2)

    raw1 = raw1[start:end]
    raw2 = raw2[start:end]

    filtered1 = filtered1[start:end]
    filtered2 = filtered2[start:end]

    # est_row = [max(est_row) - x for x in est_row[start:end]]
    # est_col = [max(est_col) - x for x in est_col[start:end]]
    #
    # row = [4 - x for x in row[start:end]]
    # col = [4 - x for x in col[start:end]]

    # print 'Accuracy for Horizontal is %.2f%% and Vertical is %.2f%%' \
    #       % (get_accuracy(est_col, col), get_accuracy(est_row, row))

    plot(figure, 211, filtered1, 'lightblue', window=len(filtered1))
    # plot(figure, 211, col, 'orange', window=len(col), twin=True)
    # plot(figure, 211, est_col, 'red', window=len(est_col), twin=True)
    # plot(figure, 211, raw1, 'blue', window=len(raw1), twin=True)

    plot(figure, 212, filtered2, 'lightgreen', window=len(filtered2))
    # plot(figure, 212, row, 'orange', window=len(row), twin=True)
    # plot(figure, 212, est_row, 'red', window=len(est_row), twin=True)
    # plot(figure, 212, raw2, 'green', window=len(raw2), twin=True)


def get_accuracy(estimate, truth, interval=5):
    checks = 0
    successes = 0
    for i in range(len(estimate)):
        if i % interval == 0:
            checks += 1
            successes += 1 if estimate[i] == truth[i] else 0
    return successes * 100.0 / checks if checks != 0 else 1


if __name__ == "__main__":
    main()
