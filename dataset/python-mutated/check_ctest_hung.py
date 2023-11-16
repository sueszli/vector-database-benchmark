import re
import sys

def escape(input):
    if False:
        while True:
            i = 10
    o = input.replace('\n', '')
    o = o.replace('\r', '')
    return o

def main():
    if False:
        while True:
            i = 10
    usage = "Usage:\n1. Download the Paddle_PR_CI_*.log from TeamCity\n2. run: python check_ctest_hung.py Paddle_PR_CI_*.log\n3. If there is hung ctest, the result likes:\nDiff:  set(['test_parallel_executor_crf'])\n    "
    if len(sys.argv) < 2:
        print(usage)
        sys.exit(0)
    logfile = sys.argv[1]
    started = set()
    passed = set()
    with open(logfile, 'r') as fn:
        for l in fn.readlines():
            if l.find('Test ') != -1 and l.find('Passed') != -1:
                m = re.search('Test\\s+#[0-9]*\\:\\s([a-z0-9_]+)', escape(l))
                passed.add(m.group(1))
            if l.find('Start ') != -1:
                start_parts = escape(l).split(' ')
                m = re.search('Start\\s+[0-9]+\\:\\s([a-z0-9_]+)', escape(l))
                started.add(m.group(1))
    print('Diff: ', started - passed)
if __name__ == '__main__':
    main()