"""
Add support for engineering notation to argparse.ArgumentParser
"""
import argparse
from gnuradio import eng_notation

def intx(string):
    if False:
        return 10
    '\n    Generic integer type, will interpret string as string literal.\n    Does the right thing for 0x1F, 0b10101, 010.\n    '
    try:
        return int(string, 0)
    except (ValueError, TypeError):
        raise argparse.ArgumentTypeError('Invalid integer value: {}'.format(string))

def eng_float(string):
    if False:
        print('Hello World!')
    '\n    Takes a string, returns a float. Accepts engineering notation.\n    Designed for use with argparse.ArgumentParser.\n    Will raise an ArgumentTypeError if not possible.\n    '
    try:
        return eng_notation.str_to_num(string)
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError('Invalid engineering notation value: {}'.format(string))