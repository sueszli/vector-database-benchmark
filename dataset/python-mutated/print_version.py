from __future__ import absolute_import
from __future__ import print_function
import os
import six
from typing import NamedTuple

class PrintVersionArgs(NamedTuple('PrintVersionArgs', [('argv0', str)])):

    def program_name(self):
        if False:
            for i in range(10):
                print('nop')
        return os.path.basename(self.argv0)

class PrintVersionAction:

    def __init__(self, out, version):
        if False:
            for i in range(10):
                print('nop')
        self.out = out
        self.version = version

    def run_action(self, args):
        if False:
            print('Hello World!')
        print_version(self.out, args.program_name(), self.version)

def print_version(out, program_name, version):
    if False:
        i = 10
        return i + 15
    print('%s %s' % (program_name, six.text_type(version)), file=out)