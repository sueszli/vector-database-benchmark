"""Simple script to be run *twice*, to check reference counting bugs.

See test_run for details."""
import sys

class C(object):

    def __init__(self, name):
        if False:
            for i in range(10):
                print('nop')
        self.name = name
        self.p = print
        self.flush_stdout = sys.stdout.flush

    def __del__(self):
        if False:
            while True:
                i = 10
        self.p('tclass.py: deleting object:', self.name)
        self.flush_stdout()
try:
    name = sys.argv[1]
except IndexError:
    pass
else:
    if name.startswith('C'):
        c = C(name)
print('ARGV 1-:', sys.argv[1:])
sys.stdout.flush()