"""Handle the SCOP DOMain file.

The DOM file has been officially deprecated. For more information see
the SCOP"release notes.":http://scop.berkeley.edu/release-notes-1.55.html
The DOM files for older releases can be found
"elsewhere at SCOP.":http://scop.mrc-lmb.cam.ac.uk/scop/parse/
"""
from .Residues import Residues

class Record:
    """Holds information for one SCOP domain.

    Attributes:
     - sid - The SCOP ID of the entry, e.g. d1anu1
     - residues - The domain definition as a Residues object
     - hierarchy - A string specifying where this domain is in the hierarchy.

    """

    def __init__(self, line=None):
        if False:
            for i in range(10):
                print('nop')
        'Initialize the class.'
        self.sid = ''
        self.residues = []
        self.hierarchy = ''
        if line:
            self._process(line)

    def _process(self, line):
        if False:
            return 10
        'Parse DOM records (PRIVATE).\n\n        Records consist of 4 tab deliminated fields;\n        sid, pdbid, residues, hierarchy\n        '
        line = line.rstrip()
        columns = line.split('\t')
        if len(columns) != 4:
            raise ValueError(f"I don't understand the format of {line}")
        (self.sid, pdbid, res, self.hierarchy) = columns
        self.residues = Residues(res)
        self.residues.pdbid = pdbid

    def __str__(self):
        if False:
            i = 10
            return i + 15
        'Represent the SCOP domain record as a tab-separated string.'
        s = []
        s.append(self.sid)
        s.append(str(self.residues).replace(' ', '\t'))
        s.append(self.hierarchy)
        return '\t'.join(s) + '\n'

def parse(handle):
    if False:
        for i in range(10):
            print('nop')
    'Iterate over a DOM file as a Dom record for each line.\n\n    Arguments:\n     - handle -- file-like object.\n\n    '
    for line in handle:
        if line.startswith('#'):
            continue
        yield Record(line)