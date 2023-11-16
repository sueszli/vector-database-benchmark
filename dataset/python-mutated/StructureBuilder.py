"""Consumer class that builds a Structure object.

This is used by the PDBParser and MMCIFparser classes.
"""
import warnings
from Bio.PDB.Structure import Structure
from Bio.PDB.Model import Model
from Bio.PDB.Chain import Chain
from Bio.PDB.Residue import Residue, DisorderedResidue
from Bio.PDB.Atom import Atom, DisorderedAtom
from Bio.PDB.PDBExceptions import PDBConstructionException
from Bio.PDB.PDBExceptions import PDBConstructionWarning

class StructureBuilder:
    """Deals with constructing the Structure object.

    The StructureBuilder class is used by the PDBParser classes to
    translate a file to a Structure object.
    """

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        'Initialize the class.'
        self.line_counter = 0
        self.header = {}

    def _is_completely_disordered(self, residue):
        if False:
            i = 10
            return i + 15
        'Return 1 if all atoms in the residue have a non blank altloc (PRIVATE).'
        atom_list = residue.get_unpacked_list()
        for atom in atom_list:
            altloc = atom.get_altloc()
            if altloc == ' ':
                return 0
        return 1

    def set_header(self, header):
        if False:
            i = 10
            return i + 15
        'Set header.'
        self.header = header

    def set_line_counter(self, line_counter):
        if False:
            return 10
        'Tracks line in the PDB file that is being parsed.\n\n        Arguments:\n         - line_counter - int\n\n        '
        self.line_counter = line_counter

    def init_structure(self, structure_id):
        if False:
            while True:
                i = 10
        'Initialize a new Structure object with given id.\n\n        Arguments:\n         - id - string\n\n        '
        self.structure = Structure(structure_id)

    def init_model(self, model_id, serial_num=None):
        if False:
            return 10
        'Create a new Model object with given id.\n\n        Arguments:\n         - id - int\n         - serial_num - int\n\n        '
        self.model = Model(model_id, serial_num)
        self.structure.add(self.model)

    def init_chain(self, chain_id):
        if False:
            i = 10
            return i + 15
        'Create a new Chain object with given id.\n\n        Arguments:\n         - chain_id - string\n\n        '
        if self.model.has_id(chain_id):
            self.chain = self.model[chain_id]
            warnings.warn('WARNING: Chain %s is discontinuous at line %i.' % (chain_id, self.line_counter), PDBConstructionWarning)
        else:
            self.chain = Chain(chain_id)
            self.model.add(self.chain)

    def init_seg(self, segid):
        if False:
            i = 10
            return i + 15
        'Flag a change in segid.\n\n        Arguments:\n         - segid - string\n\n        '
        self.segid = segid

    def init_residue(self, resname, field, resseq, icode):
        if False:
            for i in range(10):
                print('nop')
        'Create a new Residue object.\n\n        Arguments:\n         - resname - string, e.g. "ASN"\n         - field - hetero flag, "W" for waters, "H" for\n           hetero residues, otherwise blank.\n         - resseq - int, sequence identifier\n         - icode - string, insertion code\n\n        '
        if field != ' ':
            if field == 'H':
                field = 'H_' + resname
        res_id = (field, resseq, icode)
        if field == ' ':
            if self.chain.has_id(res_id):
                warnings.warn("WARNING: Residue ('%s', %i, '%s') redefined at line %i." % (field, resseq, icode, self.line_counter), PDBConstructionWarning)
                duplicate_residue = self.chain[res_id]
                if duplicate_residue.is_disordered() == 2:
                    if duplicate_residue.disordered_has_id(resname):
                        self.residue = duplicate_residue
                        duplicate_residue.disordered_select(resname)
                    else:
                        new_residue = Residue(res_id, resname, self.segid)
                        duplicate_residue.disordered_add(new_residue)
                        self.residue = duplicate_residue
                        return
                else:
                    if resname == duplicate_residue.resname:
                        warnings.warn("WARNING: Residue ('%s', %i, '%s','%s') already defined with the same name at line  %i." % (field, resseq, icode, resname, self.line_counter), PDBConstructionWarning)
                        self.residue = duplicate_residue
                        return
                    if not self._is_completely_disordered(duplicate_residue):
                        self.residue = None
                        raise PDBConstructionException("Blank altlocs in duplicate residue %s ('%s', %i, '%s')" % (resname, field, resseq, icode))
                    self.chain.detach_child(res_id)
                    new_residue = Residue(res_id, resname, self.segid)
                    disordered_residue = DisorderedResidue(res_id)
                    self.chain.add(disordered_residue)
                    disordered_residue.disordered_add(duplicate_residue)
                    disordered_residue.disordered_add(new_residue)
                    self.residue = disordered_residue
                    return
        self.residue = Residue(res_id, resname, self.segid)
        self.chain.add(self.residue)

    def init_atom(self, name, coord, b_factor, occupancy, altloc, fullname, serial_number=None, element=None, pqr_charge=None, radius=None, is_pqr=False):
        if False:
            return 10
        'Create a new Atom object.\n\n        Arguments:\n         - name - string, atom name, e.g. CA, spaces should be stripped\n         - coord - NumPy array (Float0, length 3), atomic coordinates\n         - b_factor - float, B factor\n         - occupancy - float\n         - altloc - string, alternative location specifier\n         - fullname - string, atom name including spaces, e.g. " CA "\n         - element - string, upper case, e.g. "HG" for mercury\n         - pqr_charge - float, atom charge (PQR format)\n         - radius - float, atom radius (PQR format)\n         - is_pqr - boolean, flag to specify if a .pqr file is being parsed\n\n        '
        residue = self.residue
        if residue is None:
            return
        if residue.has_id(name):
            duplicate_atom = residue[name]
            duplicate_fullname = duplicate_atom.get_fullname()
            if duplicate_fullname != fullname:
                name = fullname
                warnings.warn('Atom names %r and %r differ only in spaces at line %i.' % (duplicate_fullname, fullname, self.line_counter), PDBConstructionWarning)
        if not is_pqr:
            self.atom = Atom(name, coord, b_factor, occupancy, altloc, fullname, serial_number, element)
        elif is_pqr:
            self.atom = Atom(name, coord, None, None, altloc, fullname, serial_number, element, pqr_charge, radius)
        if altloc != ' ':
            if residue.has_id(name):
                duplicate_atom = residue[name]
                if duplicate_atom.is_disordered() == 2:
                    duplicate_atom.disordered_add(self.atom)
                else:
                    residue.detach_child(name)
                    disordered_atom = DisorderedAtom(name)
                    residue.add(disordered_atom)
                    disordered_atom.disordered_add(self.atom)
                    disordered_atom.disordered_add(duplicate_atom)
                    residue.flag_disordered()
                    warnings.warn('WARNING: disordered atom found with blank altloc before line %i.\n' % self.line_counter, PDBConstructionWarning)
            else:
                disordered_atom = DisorderedAtom(name)
                residue.add(disordered_atom)
                disordered_atom.disordered_add(self.atom)
                residue.flag_disordered()
        else:
            residue.add(self.atom)

    def set_anisou(self, anisou_array):
        if False:
            print('Hello World!')
        'Set anisotropic B factor of current Atom.'
        self.atom.set_anisou(anisou_array)

    def set_siguij(self, siguij_array):
        if False:
            return 10
        'Set standard deviation of anisotropic B factor of current Atom.'
        self.atom.set_siguij(siguij_array)

    def set_sigatm(self, sigatm_array):
        if False:
            i = 10
            return i + 15
        'Set standard deviation of atom position of current Atom.'
        self.atom.set_sigatm(sigatm_array)

    def get_structure(self):
        if False:
            while True:
                i = 10
        'Return the structure.'
        self.structure.header = self.header
        return self.structure

    def set_symmetry(self, spacegroup, cell):
        if False:
            print('Hello World!')
        'Set symmetry.'