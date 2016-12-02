import sys
from matplotlib import pyplot
import numpy as np
from scipy import signal
from pykalman import KalmanFilter

SAMPLING_RATE = 200


def main():
    filename = sys.argv[1]
    flags = sys.argv[2:]
    
    data = read(filename, 'flip' in flags)
    if 'filter' in flags:
        data = kalman(data)
    plot(data, 'grid' in flags)


def bessel(data):
    samplingRad = 1256.63706    # 200Hz
    low = 3.14159265            # 0.5Hz  30bpm
    # low = 10.47218494           # 1.6667Hz 100bpm
    high = 87.9645942           # 14Hz 840bpm
    # high = 41.889996395         # 6.667Hz 400bpm
    # high = 31.4159265         # 5Hz 300bpm
    b, a = signal.bessel(1, [low, high], btype='bandpass')
    return signal.lfilter(b, a, data)


def kalman(data):
    kf = KalmanFilter(initial_state_mean=0, n_dim_obs=1)
    kf = kf.em(data, n_iter=5)
    (filtered_state_means, filtered_state_covariances) = kf.filter(data)
    return filtered_state_means


def plot(ydata, show_grid=False):
    fig = pyplot.figure(figsize=(15, 6))
    ax = fig.add_subplot(1, 1, 1)

    # ax.set_xticklabels([])
    ax.set_yticklabels([])
    
    if show_grid:
        ax.set_xticks(np.arange(0, len(ydata), 8), minor=True)
        ax.set_xticks(np.arange(0, len(ydata), 40))

        ax.set_yticks(np.arange(0, 275, 25))
        ax.set_yticks(np.arange(0, 275, 5), minor=True)

    ax.plot(ydata, linewidth=1)
    ax.axis([0, SAMPLING_RATE * 5, 0, 275])

    ax.grid(which='both', color='r', linestyle='-')
    ax.grid(which='minor', alpha=0.2)
    ax.grid(which='major', alpha=0.5)

    pyplot.show()


def read(filename, flip=False):
    data = []
    with open(filename, 'rb') as f:
        buffer = f.read(512)
        while buffer:
            for byte in buffer:
                value = ord(byte)
                data.append(value if not flip else 255 - value)
            buffer = f.read(512)
    return data


if __name__ == "__main__":
    main()
