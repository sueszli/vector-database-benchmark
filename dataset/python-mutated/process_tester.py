"""Test program for processes."""
import os
import sys
test_file_match = 'process_test.log.*'
test_file = 'process_test.log.%d' % os.getpid()

def main() -> None:
    if False:
        i = 10
        return i + 15
    f = open(test_file, 'wb')
    stdin = sys.stdin.buffer
    stderr = sys.stderr.buffer
    stdout = sys.stdout.buffer
    b = stdin.read(4)
    f.write(b'one: ' + b + b'\n')
    stdout.write(b)
    stdout.flush()
    os.close(sys.stdout.fileno())
    b = stdin.read(4)
    f.write(b'two: ' + b + b'\n')
    stderr.write(b)
    stderr.flush()
    os.close(stderr.fileno())
    b = stdin.read(4)
    f.write(b'three: ' + b + b'\n')
    sys.exit(23)
if __name__ == '__main__':
    main()