"""Run and process output from the Wise2 package tool dnal.

Bio.Wise contains modules for running and processing the output of
some of the models in the Wise2 package by Ewan Birney available from:
ftp://ftp.ebi.ac.uk/pub/software/unix/wise2/
http://www.ebi.ac.uk/Wise2/

Bio.Wise.psw is for protein Smith-Waterman alignments
Bio.Wise.dnal is for Smith-Waterman DNA alignments
"""
import re
from subprocess import getoutput as _getoutput
from Bio import Wise
_SCORE_MATCH = 4
_SCORE_MISMATCH = -1
_SCORE_GAP_START = -5
_SCORE_GAP_EXTENSION = -1
_CMDLINE_DNAL = ['dnal', '-alb', '-nopretty']

def _build_dnal_cmdline(match, mismatch, gap, extension):
    if False:
        return 10
    res = _CMDLINE_DNAL[:]
    res.extend(['-match', str(match)])
    res.extend(['-mis', str(mismatch)])
    res.extend(['-gap', str(-gap)])
    res.extend(['-ext', str(-extension)])
    return res
_CMDLINE_FGREP_COUNT = "fgrep -c '%s' %s"

def _fgrep_count(pattern, file):
    if False:
        while True:
            i = 10
    return int(_getoutput(_CMDLINE_FGREP_COUNT % (pattern, file)))
_re_alb_line2coords = re.compile('^\\[([^:]+):[^\\[]+\\[([^:]+):')

def _alb_line2coords(line):
    if False:
        while True:
            i = 10
    return tuple((int(coord) + 1 for coord in _re_alb_line2coords.match(line).groups()))

def _get_coords(filename):
    if False:
        while True:
            i = 10
    alb = open(filename)
    start_line = None
    end_line = None
    for line in alb:
        if line.startswith('['):
            if not start_line:
                start_line = line
            else:
                end_line = line
    if end_line is None:
        return [(0, 0), (0, 0)]
    return list(zip(*map(_alb_line2coords, [start_line, end_line])))

class Statistics:
    """Calculate statistics from an ALB report."""

    def __init__(self, filename, match, mismatch, gap, extension):
        if False:
            for i in range(10):
                print('nop')
        'Initialize the class.'
        self.matches = _fgrep_count(f'"SEQUENCE" {match}', filename)
        self.mismatches = _fgrep_count(f'"SEQUENCE" {mismatch}', filename)
        self.gaps = _fgrep_count(f'"INSERT" {gap}', filename)
        if gap == extension:
            self.extensions = 0
        else:
            self.extensions = _fgrep_count(f'"INSERT" {extension}', filename)
        self.score = match * self.matches + mismatch * self.mismatches + gap * self.gaps + extension * self.extensions
        if self.matches or self.mismatches or self.gaps or self.extensions:
            self.coords = _get_coords(filename)
        else:
            self.coords = [(0, 0), (0, 0)]

    def identity_fraction(self):
        if False:
            return 10
        'Calculate the fraction of matches.'
        return self.matches / (self.matches + self.mismatches)
    header = 'identity_fraction\tmatches\tmismatches\tgaps\textensions'

    def __str__(self):
        if False:
            return 10
        'Statistics as a tab separated string.'
        return '\t'.join((str(x) for x in (self.identity_fraction(), self.matches, self.mismatches, self.gaps, self.extensions)))

def align(pair, match=_SCORE_MATCH, mismatch=_SCORE_MISMATCH, gap=_SCORE_GAP_START, extension=_SCORE_GAP_EXTENSION, **keywds):
    if False:
        for i in range(10):
            print('nop')
    'Align a pair of DNA files using dnal and calculate the statistics of the alignment.'
    cmdline = _build_dnal_cmdline(match, mismatch, gap, extension)
    temp_file = Wise.align(cmdline, pair, **keywds)
    try:
        return Statistics(temp_file.name, match, mismatch, gap, extension)
    except AttributeError:
        try:
            keywds['dry_run']
            return None
        except KeyError:
            raise

def main():
    if False:
        while True:
            i = 10
    'Command line implementation.'
    import sys
    stats = align(sys.argv[1:3])
    print('\n'.join((f'{attr}: {getattr(stats, attr)}' for attr in ('matches', 'mismatches', 'gaps', 'extensions'))))
    print(f'identity_fraction: {stats.identity_fraction()}')
    print(f'coords: {stats.coords}')

def _test(*args, **keywds):
    if False:
        return 10
    import doctest
    import sys
    doctest.testmod(sys.modules[__name__], *args, **keywds)
if __name__ == '__main__':
    if __debug__:
        _test()
    main()