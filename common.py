
'''
Created on Apr 27, 2017

@author: abhi
'''

import numpy as np


SAMPLING_RATE = 51.2


def plot(figure, row_col, data, color, x=[], max_y=0, min_y=-1, start=0, twin=False, window=SAMPLING_RATE * 60,
         bar=False):
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
        chart.bar(x, data, 0.2, linewidth=0, color=color)
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
