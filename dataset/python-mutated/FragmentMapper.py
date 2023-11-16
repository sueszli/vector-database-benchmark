"""Classify protein backbone structure with Kolodny et al's fragment libraries.

It can be regarded as a form of objective secondary structure classification.
Only fragments of length 5 or 7 are supported (ie. there is a 'central'
residue).

Full reference:

Kolodny R, Koehl P, Guibas L, Levitt M.
Small libraries of protein fragments model native protein structures accurately.
J Mol Biol. 2002 323(2):297-307.

The definition files of the fragments can be obtained from:

http://github.com/csblab/fragments/

You need these files to use this module.

The following example uses the library with 10 fragments of length 5.
The library files can be found in directory 'fragment_data'.

    >>> from Bio.PDB.PDBParser import PDBParser
    >>> from Bio.PDB.FragmentMapper import FragmentMapper
    >>> parser = PDBParser()
    >>> structure = parser.get_structure("1a8o", "PDB/1A8O.pdb")
    >>> model = structure[0]
    >>> fm = FragmentMapper(model, lsize=10, flength=5, fdir="PDB")
    >>> chain = model['A']
    >>> res152 = chain[152]
    >>> res157 = chain[157]
    >>> res152 in fm # is res152 mapped? (fragment of a C-alpha polypeptide)
    False
    >>> res157 in fm # is res157 mapped? (fragment of a C-alpha polypeptide)
    True

"""
import numpy as np
from Bio.SVDSuperimposer import SVDSuperimposer
from Bio.PDB.PDBExceptions import PDBException
from Bio.PDB.Polypeptide import PPBuilder
_FRAGMENT_FILE = 'lib_%s_z_%s.txt'

def _read_fragments(size, length, dir='.'):
    if False:
        i = 10
        return i + 15
    'Read a fragment spec file (PRIVATE).\n\n    Read a fragment spec file available from\n    http://github.com/csblab/fragments/\n    and return a list of Fragment objects.\n\n    :param size: number of fragments in the library\n    :type size: int\n\n    :param length: length of the fragments\n    :type length: int\n\n    :param dir: directory where the fragment spec files can be found\n    :type dir: string\n    '
    filename = (dir + '/' + _FRAGMENT_FILE) % (size, length)
    with open(filename) as fp:
        flist = []
        fid = 0
        for line in fp:
            if line[0] == '*' or line[0] == '\n':
                continue
            sl = line.split()
            if sl[1] == '------':
                f = Fragment(length, fid)
                flist.append(f)
                fid += 1
                continue
            coord = np.array([float(x) for x in sl[0:3]])
            f.add_residue('XXX', coord)
    return flist

class Fragment:
    """Represent a polypeptide C-alpha fragment."""

    def __init__(self, length, fid):
        if False:
            return 10
        'Initialize fragment object.\n\n        :param length: length of the fragment\n        :type length: int\n\n        :param fid: id for the fragment\n        :type fid: int\n        '
        self.length = length
        self.counter = 0
        self.resname_list = []
        self.coords_ca = np.zeros((length, 3), 'd')
        self.fid = fid

    def get_resname_list(self):
        if False:
            for i in range(10):
                print('nop')
        'Get residue list.\n\n        :return: the residue names\n        :rtype: [string, string,...]\n        '
        return self.resname_list

    def get_id(self):
        if False:
            for i in range(10):
                print('nop')
        'Get identifier for the fragment.\n\n        :return: id for the fragment\n        :rtype: int\n        '
        return self.fid

    def get_coords(self):
        if False:
            while True:
                i = 10
        'Get the CA coordinates in the fragment.\n\n        :return: the CA coords in the fragment\n        :rtype: NumPy (Nx3) array\n        '
        return self.coords_ca

    def add_residue(self, resname, ca_coord):
        if False:
            print('Hello World!')
        'Add a residue.\n\n        :param resname: residue name (eg. GLY).\n        :type resname: string\n\n        :param ca_coord: the c-alpha coordinates of the residues\n        :type ca_coord: NumPy array with length 3\n        '
        if self.counter >= self.length:
            raise PDBException('Fragment boundary exceeded.')
        self.resname_list.append(resname)
        self.coords_ca[self.counter] = ca_coord
        self.counter = self.counter + 1

    def __len__(self):
        if False:
            return 10
        'Return length of the fragment.'
        return self.length

    def __sub__(self, other):
        if False:
            print('Hello World!')
        'Return rmsd between two fragments.\n\n        :return: rmsd between fragments\n        :rtype: float\n\n        Examples\n        --------\n        This is an incomplete but illustrative example::\n\n            rmsd = fragment1 - fragment2\n\n        '
        sup = SVDSuperimposer()
        sup.set(self.coords_ca, other.coords_ca)
        sup.run()
        return sup.get_rms()

    def __repr__(self):
        if False:
            while True:
                i = 10
        'Represent the fragment object as a string.\n\n        Returns <Fragment length=L id=ID> where L=length of fragment\n        and ID the identifier (rank in the library).\n        '
        return '<Fragment length=%i id=%i>' % (self.length, self.fid)

def _make_fragment_list(pp, length):
    if False:
        i = 10
        return i + 15
    'Dice up a peptide in fragments of length "length" (PRIVATE).\n\n    :param pp: a list of residues (part of one peptide)\n    :type pp: [L{Residue}, L{Residue}, ...]\n\n    :param length: fragment length\n    :type length: int\n    '
    frag_list = []
    for i in range(len(pp) - length + 1):
        f = Fragment(length, -1)
        for j in range(length):
            residue = pp[i + j]
            resname = residue.get_resname()
            if residue.has_id('CA'):
                ca = residue['CA']
            else:
                raise PDBException('CHAINBREAK')
            if ca.is_disordered():
                raise PDBException('CHAINBREAK')
            ca_coord = ca.get_coord()
            f.add_residue(resname, ca_coord)
        frag_list.append(f)
    return frag_list

def _map_fragment_list(flist, reflist):
    if False:
        print('Hello World!')
    'Map flist fragments to closest entry in reflist (PRIVATE).\n\n    Map all frgaments in flist to the closest (in RMSD) fragment in reflist.\n\n    Returns a list of reflist indices.\n\n    :param flist: list of protein fragments\n    :type flist: [L{Fragment}, L{Fragment}, ...]\n\n    :param reflist: list of reference (ie. library) fragments\n    :type reflist: [L{Fragment}, L{Fragment}, ...]\n    '
    mapped = []
    for f in flist:
        rank = []
        for i in range(len(reflist)):
            rf = reflist[i]
            rms = f - rf
            rank.append((rms, rf))
        rank.sort()
        fragment = rank[0][1]
        mapped.append(fragment)
    return mapped

class FragmentMapper:
    """Map polypeptides in a model to lists of representative fragments."""

    def __init__(self, model, lsize=20, flength=5, fdir='.'):
        if False:
            print('Hello World!')
        'Create instance of FragmentMapper.\n\n        :param model: the model that will be mapped\n        :type model: L{Model}\n\n        :param lsize: number of fragments in the library\n        :type lsize: int\n\n        :param flength: length of fragments in the library\n        :type flength: int\n\n        :param fdir: directory where the definition files are\n                     found (default=".")\n        :type fdir: string\n        '
        if flength == 5:
            self.edge = 2
        elif flength == 7:
            self.edge = 3
        else:
            raise PDBException('Fragment length should be 5 or 7.')
        self.flength = flength
        self.lsize = lsize
        self.reflist = _read_fragments(lsize, flength, fdir)
        self.model = model
        self.fd = self._map(self.model)

    def _map(self, model):
        if False:
            for i in range(10):
                print('nop')
        'Map (PRIVATE).\n\n        :param model: the model that will be mapped\n        :type model: L{Model}\n        '
        ppb = PPBuilder()
        ppl = ppb.build_peptides(model)
        fd = {}
        for pp in ppl:
            try:
                flist = _make_fragment_list(pp, self.flength)
                mflist = _map_fragment_list(flist, self.reflist)
                for i in range(len(pp)):
                    res = pp[i]
                    if i < self.edge:
                        continue
                    elif i >= len(pp) - self.edge:
                        continue
                    else:
                        index = i - self.edge
                        assert index >= 0
                        fd[res] = mflist[index]
            except PDBException as why:
                if why == 'CHAINBREAK':
                    pass
                else:
                    raise PDBException(why) from None
        return fd

    def __contains__(self, res):
        if False:
            return 10
        'Check if the given residue is in any of the mapped fragments.\n\n        :type res: L{Residue}\n        '
        return res in self.fd

    def __getitem__(self, res):
        if False:
            i = 10
            return i + 15
        'Get an entry.\n\n        :type res: L{Residue}\n\n        :return: fragment classification\n        :rtype: L{Fragment}\n        '
        return self.fd[res]