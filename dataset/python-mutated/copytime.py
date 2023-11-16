import sys
import os
from stat import ST_ATIME, ST_MTIME

def main():
    if False:
        for i in range(10):
            print('nop')
    if len(sys.argv) != 3:
        sys.stderr.write('usage: copytime source destination\n')
        sys.exit(2)
    (file1, file2) = (sys.argv[1], sys.argv[2])
    try:
        stat1 = os.stat(file1)
    except OSError:
        sys.stderr.write(file1 + ': cannot stat\n')
        sys.exit(1)
    try:
        os.utime(file2, (stat1[ST_ATIME], stat1[ST_MTIME]))
    except OSError:
        sys.stderr.write(file2 + ': cannot change time\n')
        sys.exit(2)
if __name__ == '__main__':
    main()