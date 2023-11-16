"""Creates a text file with URLs to download FSNS dataset using aria2c.

The FSNS dataset has 640 files and takes 158Gb of the disk space. So it is
highly recommended to use some kind of a download manager to download it.

Aria2c is a powerful download manager which can download multiple files in
parallel, re-try if encounter an error and continue previously unfinished
downloads.
"""
import os
_FSNS_BASE_URL = 'http://download.tensorflow.org/data/fsns-20160927/'
_SHARDS = {'test': 64, 'train': 512, 'validation': 64}
_OUTPUT_FILE = 'fsns_urls.txt'
_OUTPUT_DIR = 'data/fsns'

def fsns_paths():
    if False:
        while True:
            i = 10
    paths = ['charset_size=134.txt']
    for (name, shards) in _SHARDS.items():
        for i in range(shards):
            paths.append('%s/%s-%05d-of-%05d' % (name, name, i, shards))
    return paths
if __name__ == '__main__':
    with open(_OUTPUT_FILE, 'w') as f:
        for path in fsns_paths():
            url = _FSNS_BASE_URL + path
            dst_path = os.path.join(_OUTPUT_DIR, path)
            f.write('%s\n  out=%s\n' % (url, dst_path))
    print('To download FSNS dataset execute:')
    print('aria2c -c -j 20 -i %s' % _OUTPUT_FILE)
    print('The downloaded FSNS dataset will be stored under %s' % _OUTPUT_DIR)