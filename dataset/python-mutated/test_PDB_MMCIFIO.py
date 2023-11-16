"""Unit tests for the Bio.PDB.MMCIFIO module."""
import os
import tempfile
import unittest
import warnings
from Bio.PDB import MMCIFParser, MMCIFIO, PDBParser, Select
from Bio.PDB import Atom, Residue
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
from Bio.PDB.PDBExceptions import PDBConstructionWarning

class WriteTest(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        if False:
            return 10
        self.io = MMCIFIO()
        self.mmcif_parser = MMCIFParser()
        self.pdb_parser = PDBParser()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', PDBConstructionWarning)
            self.structure = self.pdb_parser.get_structure('example', 'PDB/1A8O.pdb')
            self.mmcif_file = 'PDB/1A8O.cif'
            self.mmcif_multimodel_pdb_file = 'PDB/1SSU_mod.pdb'
            self.mmcif_multimodel_mmcif_file = 'PDB/1SSU_mod.cif'

    def test_mmcifio_write_structure(self):
        if False:
            for i in range(10):
                print('nop')
        'Write a full structure using MMCIFIO.'
        struct1 = self.structure
        self.io.set_structure(struct1)
        (filenumber, filename) = tempfile.mkstemp()
        os.close(filenumber)
        try:
            self.io.save(filename)
            struct2 = self.mmcif_parser.get_structure('1a8o', filename)
            nresidues = len(list(struct2.get_residues()))
            self.assertEqual(len(struct2), 1)
            self.assertEqual(nresidues, 158)
        finally:
            os.remove(filename)

    def test_mmcifio_write_residue(self):
        if False:
            return 10
        'Write a single residue using MMCIFIO.'
        struct1 = self.structure
        residue1 = list(struct1.get_residues())[0]
        self.io.set_structure(residue1)
        (filenumber, filename) = tempfile.mkstemp()
        os.close(filenumber)
        try:
            self.io.save(filename)
            struct2 = self.mmcif_parser.get_structure('1a8o', filename)
            nresidues = len(list(struct2.get_residues()))
            self.assertEqual(nresidues, 1)
        finally:
            os.remove(filename)

    def test_mmcifio_write_residue_w_chain(self):
        if False:
            for i in range(10):
                print('nop')
        'Write a single residue (chain id == X) using MMCIFIO.'
        struct1 = self.structure.copy()
        residue1 = list(struct1.get_residues())[0]
        parent = residue1.parent
        parent.id = 'X'
        self.io.set_structure(residue1)
        (filenumber, filename) = tempfile.mkstemp()
        os.close(filenumber)
        try:
            self.io.save(filename)
            struct2 = self.mmcif_parser.get_structure('1a8o', filename)
            nresidues = len(list(struct2.get_residues()))
            self.assertEqual(nresidues, 1)
            chain_id = [c.id for c in struct2.get_chains()][0]
            self.assertEqual(chain_id, 'X')
        finally:
            os.remove(filename)

    def test_mmcifio_write_residue_wout_chain(self):
        if False:
            for i in range(10):
                print('nop')
        'Write a single orphan residue using MMCIFIO.'
        struct1 = self.structure
        residue1 = list(struct1.get_residues())[0]
        residue1.parent = None
        self.io.set_structure(residue1)
        (filenumber, filename) = tempfile.mkstemp()
        os.close(filenumber)
        try:
            self.io.save(filename)
            struct2 = self.mmcif_parser.get_structure('1a8o', filename)
            nresidues = len(list(struct2.get_residues()))
            self.assertEqual(nresidues, 1)
            chain_id = [c.id for c in struct2.get_chains()][0]
            self.assertEqual(chain_id, 'A')
        finally:
            os.remove(filename)

    def test_mmcifio_write_custom_residue(self):
        if False:
            while True:
                i = 10
        'Write a chainless residue using PDBIO.'
        res = Residue.Residue((' ', 1, ' '), 'DUM', '')
        atm = Atom.Atom('CA', [0.1, 0.1, 0.1], 1.0, 1.0, ' ', 'CA', 1, 'C')
        res.add(atm)
        parent = res.parent
        self.io.set_structure(res)
        self.assertIs(parent, res.parent)
        (filenumber, filename) = tempfile.mkstemp()
        os.close(filenumber)
        try:
            self.io.save(filename)
            struct2 = self.mmcif_parser.get_structure('res', filename)
            latoms = list(struct2.get_atoms())
            self.assertEqual(len(latoms), 1)
            self.assertEqual(latoms[0].name, 'CA')
            self.assertEqual(latoms[0].parent.resname, 'DUM')
            self.assertEqual(latoms[0].parent.parent.id, 'A')
        finally:
            os.remove(filename)

    def test_mmcifio_select(self):
        if False:
            return 10
        'Write a selection of the structure using a Select subclass.'

        class CAonly(Select):
            """Accepts only CA residues."""

            def accept_atom(self, atom):
                if False:
                    i = 10
                    return i + 15
                if atom.name == 'CA' and atom.element == 'C':
                    return 1
        struct1 = self.structure
        self.io.set_structure(struct1)
        (filenumber, filename) = tempfile.mkstemp()
        os.close(filenumber)
        try:
            self.io.save(filename, CAonly())
            struct2 = self.mmcif_parser.get_structure('1a8o', filename)
            nresidues = len(list(struct2.get_residues()))
            self.assertEqual(nresidues, 70)
        finally:
            os.remove(filename)

    def test_mmcifio_write_dict(self):
        if False:
            return 10
        'Write an mmCIF dictionary out, read it in and compare them.'
        d1 = MMCIF2Dict(self.mmcif_file)
        self.io.set_dict(d1)
        (filenumber, filename) = tempfile.mkstemp()
        os.close(filenumber)
        try:
            self.io.save(filename)
            d2 = MMCIF2Dict(filename)
            k1 = sorted(d1.keys())
            k2 = sorted(d2.keys())
            self.assertEqual(k1, k2)
            for key in k1:
                self.assertEqual(d1[key], d2[key])
        finally:
            os.remove(filename)

    def test_mmcifio_multimodel(self):
        if False:
            while True:
                i = 10
        'Write a multi-model, multi-chain mmCIF file.'
        pdb_struct = self.pdb_parser.get_structure('1SSU_mod_pdb', self.mmcif_multimodel_pdb_file)
        mmcif_struct = self.mmcif_parser.get_structure('1SSU_mod_mmcif', self.mmcif_multimodel_mmcif_file)
        io = MMCIFIO()
        for struct in [pdb_struct, mmcif_struct]:
            self.io.set_structure(struct)
            (filenumber, filename) = tempfile.mkstemp()
            os.close(filenumber)
            try:
                self.io.save(filename)
                struct_in = self.mmcif_parser.get_structure('1SSU_mod_in', filename)
                self.assertEqual(len(struct_in), 2)
                self.assertEqual(len(struct_in[1]), 2)
                self.assertAlmostEqual(struct_in[1]['B'][1]['N'].get_coord()[0], 6.259, 3)
            finally:
                os.remove(filename)
if __name__ == '__main__':
    runner = unittest.TextTestRunner(verbosity=2)
    unittest.main(testRunner=runner)