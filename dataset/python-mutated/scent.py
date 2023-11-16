from sniffer.api import *
import os
watch_paths = ['.', 'koans/']

@file_validator
def py_files(filename):
    if False:
        i = 10
        return i + 15
    return filename.endswith('.py') and (not os.path.basename(filename).startswith('.'))

@runnable
def execute_koans(*args):
    if False:
        while True:
            i = 10
    os.system('python3 -B contemplate_koans.py')