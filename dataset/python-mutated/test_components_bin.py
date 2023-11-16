"""Python Fire test components Fire CLI.

This file is useful for replicating test results manually.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import fire
from fire import test_components

def main():
    if False:
        while True:
            i = 10
    fire.Fire(test_components)
if __name__ == '__main__':
    main()