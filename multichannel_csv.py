'''
Created on Dec 14, 2016

@author: abhi
'''

import csv
import struct
import sys


def main():
    filename = sys.argv[1]
    rows = read(filename)
    with open(filename + '.csv', 'wb') as f:
        writer = csv.writer(f)
        writer.writerow(['time_millis', 'channel1', 'channel2'])
        writer.writerows(rows)


def read(filename):
    rows = []
    ts = 0
    with open(filename, 'rb') as f:
        buff = f.read(512)
        while buff:
            ints = list(buff[i:i + 4] for i in xrange(0, len(buff), 4))
            for value1, value2 in zip(ints[0::2], ints[1::2]):
                rows.append([ts, parse_int(value1), parse_int(value2)])
                ts += 5
            buff = f.read(512)
    return rows


def parse_int(value):
    try:
        return struct.unpack("!L", value)[0]
    except struct.error:
        return -1


if __name__ == "__main__":
    main()
