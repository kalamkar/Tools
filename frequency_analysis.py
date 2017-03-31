'''
Created on Dec 02, 2016

@author: abhi
'''

import struct
import sys
import scipy
import scipy.signal
import pylab


SAMPLING_RATE = 200


def main():
    filename = sys.argv[1]
    channel1, channel2 = read(filename)

    freqs, power = scipy.signal.periodogram(channel2, SAMPLING_RATE)

    pylab.subplot(211)
    pylab.plot(range(0, len(channel2)), channel2)
    pylab.axis([0, len(channel2), 0.3 * 1e7, 1.0 * 1e7])
    pylab.subplot(212)
    pylab.plot(freqs, power)
    pylab.axis([0, 10, 0, 1e12])

    pylab.show()


def read(filename):
    channel1 = []
    channel2 = []
    with open(filename, 'rb') as f:
        buff = f.read(512)
        while buff:
            ints = list(buff[i:i + 4] for i in xrange(0, len(buff), 4))
            for value in ints[::2]:
                channel1.append(parse_int(value))
            for value in ints[1::2]:
                channel2.append(parse_int(value))
            buff = f.read(512)
    return channel1, channel2


def parse_int(value):
    try:
        return struct.unpack("!L", value)[0]
    except struct.error:
        return -1


if __name__ == "__main__":
    main()
