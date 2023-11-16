from __future__ import annotations
import re

def read_file(path):
    if False:
        for i in range(10):
            print('nop')
    try:
        with open(path, 'r') as f:
            return f.read()
    except Exception as ex:
        print('%s:%d:%d: unable to read required file %s' % (path, 0, 0, re.sub('\\s+', ' ', str(ex))))
        return None

def main():
    if False:
        i = 10
        return i + 15
    ORIGINAL_FILE = 'requirements.txt'
    VENDORED_COPY = 'test/lib/ansible_test/_data/requirements/ansible.txt'
    original_requirements = read_file(ORIGINAL_FILE)
    vendored_requirements = read_file(VENDORED_COPY)
    if original_requirements is not None and vendored_requirements is not None:
        if original_requirements != vendored_requirements:
            print('%s:%d:%d: must be identical to %s' % (VENDORED_COPY, 0, 0, ORIGINAL_FILE))
if __name__ == '__main__':
    main()