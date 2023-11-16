"""A collection of residues from a PDB structure."""
import re
_pdbid_re = re.compile('^(\\w\\w\\w\\w)(?:$|\\s+|_)(.*)')
_fragment_re = re.compile('\\(?(\\w:)?(-?\\w*)-?(-?\\w*)\\)?(.*)')

class Residues:
    """A collection of residues from a PDB structure.

    This class provides code to work with SCOP domain definitions. These
    are concisely expressed as one or more chain fragments. For example,
    "(1bba A:10-20,B:)" indicates residue 10 through 20 (inclusive) of
    chain A, and every residue of chain B in the pdb structure 1bba. The pdb
    id and brackets are optional. In addition "-" indicates every residue of
    a pbd structure with one unnamed chain.

    Start and end residue ids consist of the residue sequence number and an
    optional single letter insertion code. e.g. "12", "-1", "1a", "1000"


    pdbid -- An optional PDB id, e.g. "1bba"

    fragments -- A sequence of tuples (chainID, startResID, endResID)

    """

    def __init__(self, str=None):
        if False:
            while True:
                i = 10
        'Initialize the class.'
        self.pdbid = ''
        self.fragments = ()
        if str is not None:
            self._parse(str)

    def _parse(self, str):
        if False:
            while True:
                i = 10
        str = str.strip()
        m = _pdbid_re.match(str)
        if m is not None:
            self.pdbid = m.group(1)
            str = m.group(2)
        if str == '' or str == '-' or str == '(-)':
            return
        fragments = []
        for fragment in str.split(','):
            m = _fragment_re.match(fragment)
            if m is None:
                raise ValueError(f"I don't understand the format of {fragment}")
            (chain, start, end, postfix) = m.groups()
            if postfix != '':
                raise ValueError(f"I don't understand the format of {fragment}")
            if chain:
                if chain[-1] != ':':
                    raise ValueError(f"I don't understand the chain in {fragment}")
                chain = chain[:-1]
            else:
                chain = ''
            fragments.append((chain, start, end))
        self.fragments = tuple(fragments)

    def __str__(self):
        if False:
            print('Hello World!')
        'Represent the SCOP residues record as a string.'
        prefix = ''
        if self.pdbid:
            prefix = self.pdbid + ' '
        if not self.fragments:
            return prefix + '-'
        strs = []
        for (chain, start, end) in self.fragments:
            s = []
            if chain:
                s.append(f'{chain}:')
            if start:
                s.append(f'{start}-{end}')
            strs.append(''.join(s))
        return prefix + ','.join(strs)