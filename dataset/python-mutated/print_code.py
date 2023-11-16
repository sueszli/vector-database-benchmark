"""
Created on May 10, 2012

@author: sean
"""
from __future__ import print_function as _
from __future__ import division as _
from __future__ import absolute_import as _
from .bytecode_consumer import ByteCodeConsumer
from argparse import ArgumentParser

class ByteCodePrinter(ByteCodeConsumer):

    def generic_consume(self, instr):
        if False:
            while True:
                i = 10
        print(instr)

def main():
    if False:
        while True:
            i = 10
    parser = ArgumentParser()
    parser.add_argument()
if __name__ == '__main__':
    main()