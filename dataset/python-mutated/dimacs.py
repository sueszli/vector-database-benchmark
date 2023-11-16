"""For reading in DIMACS file format

www.cs.ubc.ca/~hoos/SATLIB/Benchmarks/SAT/satformat.ps

"""
from sympy.core import Symbol
from sympy.logic.boolalg import And, Or
import re

def load(s):
    if False:
        for i in range(10):
            print('nop')
    "Loads a boolean expression from a string.\n\n    Examples\n    ========\n\n    >>> from sympy.logic.utilities.dimacs import load\n    >>> load('1')\n    cnf_1\n    >>> load('1 2')\n    cnf_1 | cnf_2\n    >>> load('1 \\n 2')\n    cnf_1 & cnf_2\n    >>> load('1 2 \\n 3')\n    cnf_3 & (cnf_1 | cnf_2)\n    "
    clauses = []
    lines = s.split('\n')
    pComment = re.compile('c.*')
    pStats = re.compile('p\\s*cnf\\s*(\\d*)\\s*(\\d*)')
    while len(lines) > 0:
        line = lines.pop(0)
        if not pComment.match(line):
            m = pStats.match(line)
            if not m:
                nums = line.rstrip('\n').split(' ')
                list = []
                for lit in nums:
                    if lit != '':
                        if int(lit) == 0:
                            continue
                        num = abs(int(lit))
                        sign = True
                        if int(lit) < 0:
                            sign = False
                        if sign:
                            list.append(Symbol('cnf_%s' % num))
                        else:
                            list.append(~Symbol('cnf_%s' % num))
                if len(list) > 0:
                    clauses.append(Or(*list))
    return And(*clauses)

def load_file(location):
    if False:
        while True:
            i = 10
    'Loads a boolean expression from a file.'
    with open(location) as f:
        s = f.read()
    return load(s)