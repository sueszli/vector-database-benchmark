import os
import sys

def sort_by_dict_order(file_path):
    if False:
        i = 10
        return i + 15
    with open(file_path, 'r') as f:
        lines = f.readlines()
    sorted_lines = sorted(lines)
    with open(file_path, 'w') as f:
        f.writelines(sorted_lines)
if __name__ == '__main__':
    file_paths = sys.argv[1:]
    for file_path in file_paths:
        file_path = os.path.normpath(file_path)
        sort_by_dict_order(file_path)