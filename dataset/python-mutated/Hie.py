"""Handle the SCOP HIErarchy files.

The SCOP Hierarchy files describe the SCOP hierarchy in terms of SCOP
unique identifiers (sunid).

The file format is described in the SCOP `release notes
<http://scop.berkeley.edu/release-notes-1.55.html>`_.

The latest HIE file can be found `elsewhere at SCOP
<http://scop.mrc-lmb.cam.ac.uk/scop/parse/>`_.

`Release 1.55 <http://scop.berkeley.edu/parse/dir.hie.scop.txt_1.55>`_
(July 2001).
"""

class Record:
    """Holds information for one node in the SCOP hierarchy.

    Attributes:
     - sunid - SCOP unique identifiers of this node
     - parent - Parents sunid
     - children - Sequence of children sunids

    """

    def __init__(self, line=None):
        if False:
            for i in range(10):
                print('nop')
        'Initialize the class.'
        self.sunid = ''
        self.parent = ''
        self.children = []
        if line:
            self._process(line)

    def _process(self, line):
        if False:
            for i in range(10):
                print('nop')
        "Parse HIE records (PRIVATE).\n\n        Records consist of 3 tab deliminated fields; node's sunid,\n        parent's sunid, and a list of children's sunids.\n        "
        line = line.rstrip()
        columns = line.split('\t')
        if len(columns) != 3:
            raise ValueError(f"I don't understand the format of {line}")
        (sunid, parent, children) = columns
        if sunid == '-':
            self.sunid = ''
        else:
            self.sunid = int(sunid)
        if parent == '-':
            self.parent = ''
        else:
            self.parent = int(parent)
        if children == '-':
            self.children = ()
        else:
            children = children.split(',')
            self.children = [int(x) for x in children]

    def __str__(self):
        if False:
            for i in range(10):
                print('nop')
        'Represent the SCOP hierarchy record as a string.'
        s = []
        s.append(str(self.sunid))
        if self.parent:
            s.append(str(self.parent))
        elif self.sunid != 0:
            s.append('0')
        else:
            s.append('-')
        if self.children:
            s.append(','.join((str(x) for x in self.children)))
        else:
            s.append('-')
        return '\t'.join(s) + '\n'

def parse(handle):
    if False:
        for i in range(10):
            print('nop')
    'Iterate over a HIE file as Hie records for each line.\n\n    Arguments:\n     - handle - file-like object.\n\n    '
    for line in handle:
        if line.startswith('#'):
            continue
        yield Record(line)