
'''
Created on Apr 27, 2017

@author: abhi
'''

import numpy as np


SAMPLING_RATE = 51.2


def plot(figure, row_col, data, color, x=[], max_y=0, min_y=-1, start=0, twin=False, window=SAMPLING_RATE * 60,
         bar=False, yerr=None):
    if len(data) == 0:
        return

    chart = figure.add_subplot(row_col)
    if twin:
        chart = chart.twinx()
    chart.set_xticks(np.arange(0, len(data), window / 15))
    if len(x) > 0:
        chart.scatter(x, data, linewidth=1, color=color)
    elif bar:
        x = range(0, len(data))
        if yerr:
            chart.bar(x, data, 0.8, linewidth=0, color=color, yerr=yerr)
        else:
            chart.bar(x, data, 0.8, linewidth=0, color=color)
    else:
        chart.plot(data, linewidth=1, color=color)

    chart.set_xbound([start, start + window])
    if max_y and min_y == -1:
        chart.set_ylim([-max_y, max_y])
    elif max_y:
        chart.set_ylim([min_y, max_y])

    chart.margins(0.0, 0.05)
    chart.grid(not twin)


def get_slope(data):
    if len(data) < 2:
        return 0
    median_size = max(1, len(data) / 20)  # 5%

    start = int(np.median(data[:median_size]))
    end = int(np.median(data[-median_size:]))
    return float((end - start) / (len(data) - 1))


def is_peak_or_valley(values, median=0):
    if len(values) < 3:
        return False

    start = values[-3]
    middle = values[-2]
    end = values[-1]

    # All 3 points need to be on one side of the median (either all above or all below)
    all_below = (start < median and middle < median and end < median)
    all_above = (start > median and middle > median and end > median)
    peak = (start < middle and middle > end)
    valley = (start > middle and middle < end)
    return (all_below or all_above) and (peak or valley)