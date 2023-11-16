"""Python file with invalid syntax, used by scripts/linters/
python_linter_test. This file contain non alphabetical import order.
"""
from __future__ import annotations
import argparse
import fnmatch
import multiprocessing
import os
import threading
import subprocess
import sys

def fun():
    if False:
        for i in range(10):
            print('nop')
    'Function docstring.'
    print('hello')