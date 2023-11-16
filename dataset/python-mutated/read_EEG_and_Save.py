import sys
from pyOpenBCI import OpenBCICyton
import csv
import os.path
import time
import utils
start_time = time.time()
time_limit = 30
label = 'light_onn'

def print_raw(sample):
    if False:
        for i in range(10):
            print('nop')
    if os.path.exists('test.csv'):
        csvfile = open('test.csv', 'a')
        writer = csv.writer(csvfile)
    else:
        csvfile = open('test.csv', 'w')
        writer = csv.writer(csvfile)
        writer.writerow(['timestamp', 'channel1', 'channel2', 'channel3', 'channel4', 'channel5', 'channel6', 'channel7', 'channel8', 'label'])
    timestamp = int(time.time() * 1000)
    writer.writerow([timestamp] + sample.channels_data + [label])
    csvfile.close()
    print(sample.channels_data)
    if time.time() - start_time >= time_limit:
        board.stop_stream()
        sys.exit()
board = OpenBCICyton(port='COM4', daisy=False)
board.start_stream(print_raw)