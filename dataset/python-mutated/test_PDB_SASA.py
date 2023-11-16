"""Unit tests for the Bio.PDB.SASA module: Surface Accessibility Calculations."""
import copy
import pathlib
import unittest
import warnings
from Bio.PDB import PDBParser
from Bio.PDB.SASA import ShrakeRupley
DATADIR = pathlib.Path(__file__).parent / 'PDB'

class TestShrakeRupley(unittest.TestCase):
    """Tests for SR algorithm."""

    @classmethod
    def setUpClass(cls):
        if False:
            i = 10
            return i + 15
        'One-time setup for all tests.'
        cls.parser = p = PDBParser(QUIET=1)
        with warnings.catch_warnings():
            structure = p.get_structure('X', DATADIR / '1LCD.pdb')
            model = structure[0]
            for r in list(model.get_residues()):
                if r.id[0] == ' ':
                    for a in list(r):
                        if a.element == 'H':
                            r.detach_child(a.name)
                else:
                    c = r.parent
                    c.detach_child(r.id)
            cls.model = model

    def test_default_algorithm(self):
        if False:
            return 10
        'Run Shrake-Rupley with default parameters.'
        m = copy.deepcopy(self.model)
        sasa = ShrakeRupley()
        sasa.compute(m)
        result = [a.sasa for a in m.get_atoms()][:5]
        expected = [50.36, 31.4, 10.87, 12.86, 2.42]
        for (a, b) in zip(result, expected):
            self.assertAlmostEqual(a, b, places=2)

    def test_higher_resolution(self):
        if False:
            for i in range(10):
                print('nop')
        'Run Shrake-Rupley with 960 points per sphere.'
        m = copy.deepcopy(self.model)
        sasa = ShrakeRupley(n_points=960)
        sasa.compute(m)
        result = [a.sasa for a in m.get_atoms()][:5]
        expected = [51.9, 31.45, 12.45, 12.72, 3.02]
        for (a, b) in zip(result, expected):
            self.assertAlmostEqual(a, b, places=2)

    def test_custom_radii(self):
        if False:
            return 10
        'Run Shrake-Rupley with custom radii.'
        m = copy.deepcopy(self.model)
        sasa = ShrakeRupley(radii_dict={'C': 5.0})
        sasa.compute(m)
        result = [a.sasa for a in m.get_atoms()][:5]
        expected = [0.0, 190.45, 41.18, 0.0, 36.03]
        for (a, b) in zip(result, expected):
            self.assertAlmostEqual(a, b, places=2)

    def test_level_R(self):
        if False:
            i = 10
            return i + 15
        'Run Shrake-Rupley with level R.'
        m = copy.deepcopy(self.model)
        sasa = ShrakeRupley()
        sasa.compute(m, level='R')
        for r in m.get_residues():
            atom_sum = sum((a.sasa for a in r))
            self.assertAlmostEqual(atom_sum, r.sasa, places=2)

    def test_level_C(self):
        if False:
            i = 10
            return i + 15
        'Run Shrake-Rupley with level C.'
        m = copy.deepcopy(self.model)
        sasa = ShrakeRupley()
        sasa.compute(m, level='C')
        for c in m.get_chains():
            atom_sum = sum((a.sasa for a in c.get_atoms()))
            self.assertAlmostEqual(atom_sum, c.sasa, places=2)

    def test_fail_probe_radius(self):
        if False:
            for i in range(10):
                print('nop')
        'Raise exception on bad probe_radius parameter.'
        with self.assertRaisesRegex(ValueError, 'must be a positive number'):
            sasa = ShrakeRupley(probe_radius=-1.4)

    def test_fail_n_points(self):
        if False:
            return 10
        'Raise exception on bad n_points parameter.'
        with self.assertRaisesRegex(ValueError, 'must be larger than 1'):
            sasa = ShrakeRupley(n_points=0)

    def test_fail_compute_entity_type(self):
        if False:
            print('Hello World!')
        'Raise exception on unsupported entity type.'
        with self.assertRaisesRegex(ValueError, 'Invalid entity type'):
            sasa = ShrakeRupley()
            sasa.compute([1, 2, 3, 4, 5])

    def test_fail_compute_entity_level(self):
        if False:
            i = 10
            return i + 15
        'Raise exception on input Atom entity.'
        atom = list(self.model.get_atoms())[0]
        with self.assertRaisesRegex(ValueError, 'Invalid entity type'):
            sasa = ShrakeRupley()
            sasa.compute(atom)

    def test_fail_compute_level_1(self):
        if False:
            return 10
        'Raise exception on invalid level parameter: X.'
        with self.assertRaisesRegex(ValueError, 'Invalid level'):
            sasa = ShrakeRupley()
            sasa.compute(self.model, level='X')

    def test_fail_compute_level_2(self):
        if False:
            for i in range(10):
                print('nop')
        'Raise exception on invalid level parameter: S > C.'
        chain = self.model['A']
        with self.assertRaisesRegex(ValueError, 'be equal or smaller than'):
            sasa = ShrakeRupley()
            sasa.compute(chain, level='S')

    def test_fail_empty_entity(self):
        if False:
            for i in range(10):
                print('nop')
        'Raise exception on invalid level parameter: S > C.'
        sasa = ShrakeRupley()
        r = copy.deepcopy(self.model['A'].child_list[0])
        for a in list(r):
            r.detach_child(a.name)
        self.assertEqual(len(r.child_list), 0)
        with self.assertRaisesRegex(ValueError, 'Entity has no child atoms'):
            sasa.compute(r)
if __name__ == '__main__':
    runner = unittest.TextTestRunner(verbosity=2)
    unittest.main(testRunner=runner)