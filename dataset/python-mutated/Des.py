"""Handle the SCOP DEScription file.

The file format is described in the scop
"release notes.":http://scop.berkeley.edu/release-notes-1.55.html
The latest DES file can be found
"elsewhere at SCOP.":http://scop.mrc-lmb.cam.ac.uk/scop/parse/

"Release 1.55":http://scop.berkeley.edu/parse/des.cla.scop.txt_1.55 (July 2001)
"""

class Record:
    """Holds information for one node in the SCOP hierarchy.

    Attributes:
     - sunid - SCOP unique identifiers
     - nodetype - One of 'cl' (class), 'cf' (fold), 'sf' (superfamily),
       'fa' (family), 'dm' (protein), 'sp' (species), 'px' (domain).
       Additional node types may be added.
     - sccs - SCOP concise classification strings. e.g. b.1.2.1
     - name - The SCOP ID (sid) for domains (e.g. d1anu1), currently empty for other node types
     - description - e.g. "All beta proteins","Fibronectin type III",

    """

    def __init__(self, line=None):
        if False:
            for i in range(10):
                print('nop')
        'Initialize the class.'
        self.sunid = ''
        self.nodetype = ''
        self.sccs = ''
        self.name = ''
        self.description = ''
        if line:
            self._process(line)

    def _process(self, line):
        if False:
            return 10
        'Parse DES records (PRIVATE).\n\n        Records consist of 5 tab deliminated fields,\n        sunid, node type, sccs, node name, node description.\n        '
        line = line.rstrip()
        columns = line.split('\t')
        if len(columns) != 5:
            raise ValueError(f"I don't understand the format of {line}")
        (sunid, self.nodetype, self.sccs, self.name, self.description) = columns
        if self.name == '-':
            self.name = ''
        self.sunid = int(sunid)

    def __str__(self):
        if False:
            print('Hello World!')
        'Represent the SCOP description record as a tab-separated string.'
        s = []
        s.append(self.sunid)
        s.append(self.nodetype)
        s.append(self.sccs)
        if self.name:
            s.append(self.name)
        else:
            s.append('-')
        s.append(self.description)
        return '\t'.join(map(str, s)) + '\n'

def parse(handle):
    if False:
        for i in range(10):
            print('nop')
    'Iterate over a DES file as a Des record for each line.\n\n    Arguments:\n     - handle - file-like object\n\n    '
    for line in handle:
        if line.startswith('#'):
            continue
        yield Record(line)