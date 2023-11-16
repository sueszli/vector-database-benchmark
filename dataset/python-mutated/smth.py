import sys
sys.path.append('/foo/smth.py:module')

def extend_path_foo():
    if False:
        i = 10
        return i + 15
    sys.path.append('/foo/smth.py:from_func')