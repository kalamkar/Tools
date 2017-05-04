'''
Created on Dec 02, 2016

@author: abhi
'''

import csv
import sys
import time

import scipy.signal
import scipy.stats
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
    raw1 = np.array(columns[2][1:]).astype(np.int)
    raw2 = np.array(columns[3][1:]).astype(np.int)

    [filtered1, markers11, markers12, markers13], \
    [filtered2, markers21, markers22, markers23] = process(raw1, raw2)
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

    markers13 = markers13[start:end]
    markers23 = markers23[start:end]

    # plot(figure, 211, raw1, 'lightblue', window=len(raw1), twin=True)
    plot(figure, 211, filtered1, 'blue', window=len(filtered1))
    plot(figure, 211, markers11, 'orange', window=len(markers11), twin=True, bar=True, yerr=markers13)
    plot(figure, 211, markers12, 'yellow', window=len(markers12))

    # plot(figure, 212, raw2, 'lightgreen', window=len(raw2), twin=True)
    plot(figure, 212, filtered2, 'green', window=len(filtered2))
    plot(figure, 212, markers21, 'orange', window=len(markers21), twin=True, bar=True, yerr=markers23)
    plot(figure, 212, markers22, 'yellow', window=len(markers22))


def process(horizontal, vertical):
    h_filtered = []
    v_filtered = []

    gestures = VariableLengthGestureRecognizer()

    for i in range(0, len(horizontal)):

        h_value = horizontal[i]
        v_value = vertical[i]

        gestures.update(h_value, v_value)

        h_filtered.append(h_value)
        v_filtered.append(v_value)

    return [h_filtered, gestures.h_saccades, [], gestures.h_lengths], \
           [v_filtered, gestures.v_saccades, [], gestures.v_lengths]


class VariableLengthGestureRecognizer:
    def __init__(self):
        self.h_prev_value = 0
        self.v_prev_value = 0

        self.h_direction = 0
        self.v_direction = 0

        self.h_length = 0
        self.v_length = 0

        self.h_latest_change_value = 0
        self.v_latest_change_value = 0

        self.h_saccades = []
        self.v_saccades = []

        self.h_lengths = []
        self.v_lengths = []

    def update(self, horizontal, vertical):
        h_new_direction = horizontal - self.h_prev_value
        h_new_direction /= abs(h_new_direction) if h_new_direction != 0 else 1
        v_new_direction = vertical - self.v_prev_value
        v_new_direction /= abs(v_new_direction) if v_new_direction != 0 else 1

        length = 0
        if self.h_direction != h_new_direction and h_new_direction != 0:
            length = self.h_length
            self.h_length = 0
            self.h_direction = h_new_direction
        elif self.v_direction != v_new_direction and v_new_direction != 0:
            length = self.v_length
            self.v_length = 0
            self.v_direction = v_new_direction
        else:
            self.h_length += 1
            self.v_length += 1

        if length > 0:
            h_amplitude = self.h_prev_value - self.h_latest_change_value
            v_amplitude = self.v_prev_value - self.v_latest_change_value

            self.h_latest_change_value = self.h_prev_value
            self.v_latest_change_value = self.v_prev_value

            self.h_saccades.append(h_amplitude)
            self.v_saccades.append(v_amplitude)

            self.h_lengths.append(length**2)
            self.v_lengths.append(length**2)
        else:
            self.h_saccades.append(0)
            self.v_saccades.append(0)
            self.h_lengths.append(0)
            self.v_lengths.append(0)

        self.h_prev_value = horizontal
        self.v_prev_value = vertical


if __name__ == "__main__":
    main()
