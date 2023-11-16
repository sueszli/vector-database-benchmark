"""Custom entry-point for pip that filters out unwanted logging and warnings."""
from __future__ import annotations
import logging
import os
import re
import runpy
import sys
BUILTIN_FILTERER_FILTER = logging.Filterer.filter
LOGGING_MESSAGE_FILTER = re.compile("^(.*Running pip install with root privileges is generally not a good idea.*|.*Running pip as the 'root' user can result in broken permissions .*|Ignoring .*: markers .* don't match your environment|Looking in indexes: .*|Requirement already satisfied.*)$")

def custom_filterer_filter(self, record):
    if False:
        for i in range(10):
            print('nop')
    'Globally omit logging of unwanted messages.'
    if LOGGING_MESSAGE_FILTER.search(record.getMessage()):
        return 0
    return BUILTIN_FILTERER_FILTER(self, record)

def main():
    if False:
        return 10
    'Main program entry point.'
    logging.Filterer.filter = custom_filterer_filter
    get_pip = os.environ.get('GET_PIP')
    try:
        if get_pip:
            (directory, filename) = os.path.split(get_pip)
            module = os.path.splitext(filename)[0]
            sys.path.insert(0, directory)
            runpy.run_module(module, run_name='__main__', alter_sys=True)
        else:
            runpy.run_module('pip.__main__', run_name='__main__', alter_sys=True)
    except ImportError as ex:
        print('pip is unavailable: %s' % ex)
        sys.exit(1)
if __name__ == '__main__':
    main()