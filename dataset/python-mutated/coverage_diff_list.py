"""
usage: coverage_diff_list.py list_file max_rate > coverage-diff-list-90.out
"""
import sys

def filter_by(list_file, max_rate):
    if False:
        while True:
            i = 10
    '\n    Args:\n        list_file (str): File of list.\n        max_rate (float): Max rate.\n\n    Returns:\n        tuple: File and coverage rate.\n    '
    with open(list_file) as list_file:
        for line in list_file:
            line = line.strip()
            split = line.split('|')
            name = split[0].strip()
            if name.startswith('/paddle/'):
                name = name[len('/paddle/'):]
            try:
                rate = split[1].split()[0].strip('%')
                rate = float(rate)
                if rate >= max_rate:
                    continue
            except:
                pass
            print(name, rate)
if __name__ == '__main__':
    if len(sys.argv) < 2:
        sys.exit()
    list_file = sys.argv[1]
    max_rate = float(sys.argv[2])
    filter_by(list_file, max_rate)