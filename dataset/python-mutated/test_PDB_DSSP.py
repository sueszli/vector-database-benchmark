"""Unit tests for the Bio.PDB.DSSP submodule."""
import re
import subprocess
import unittest
import warnings
try:
    import numpy
except ImportError:
    from Bio import MissingPythonDependencyError
    raise MissingPythonDependencyError('Install NumPy if you want to use Bio.PDB.') from None
from Bio.PDB import PDBParser, MMCIFParser
from Bio.PDB import DSSP, make_dssp_dict
VERSION_2_2_0 = (2, 2, 0)

def parse_dssp_version(version_string):
    if False:
        return 10
    'Parse the DSSP version into a tuple from the tool output.'
    match = re.search('\\s*([\\d.]+)', version_string)
    if match:
        version = match.group(1)
    return tuple(map(int, version.split('.')))

def will_it_float(s):
    if False:
        for i in range(10):
            print('nop')
    'Convert the input into a float if it is a number.\n\n    If the input is a string, the output does not change.\n    '
    try:
        return float(s)
    except ValueError:
        return s

class DSSP_tool_test(unittest.TestCase):
    """Test calling DSSP from Bio.PDB."""

    @classmethod
    def setUpClass(cls):
        if False:
            return 10
        cls.dssp_version = (0, 0, 0)
        is_dssp_available = False
        quiet_kwargs = {'stdout': subprocess.PIPE, 'stderr': subprocess.STDOUT}
        try:
            try:
                version_string = subprocess.check_output(['dssp', '--version'], text=True)
                cls.dssp_version = parse_dssp_version(version_string)
                is_dssp_available = True
            except subprocess.CalledProcessError:
                subprocess.check_call(['dssp', '-h'], **quiet_kwargs)
                is_dssp_available = True
        except OSError:
            try:
                version_string = subprocess.check_output(['mkdssp', '--version'], text=True)
                cls.dssp_version = parse_dssp_version(version_string)
                is_dssp_available = True
            except OSError:
                pass
        if not is_dssp_available:
            raise unittest.SkipTest('Install dssp if you want to use it from Biopython.')
        cls.pdbparser = PDBParser()
        cls.cifparser = MMCIFParser()

    def test_dssp(self):
        if False:
            i = 10
            return i + 15
        'Test DSSP generation from PDB.'
        pdbfile = 'PDB/2BEG.pdb'
        model = self.pdbparser.get_structure('2BEG', pdbfile)[0]
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            dssp = DSSP(model, pdbfile)
        self.assertEqual(len(dssp), 130)

    def test_dssp_with_mmcif_file(self):
        if False:
            return 10
        'Test DSSP generation from MMCIF.'
        if self.dssp_version < VERSION_2_2_0:
            self.skipTest('Test requires DSSP version 2.2.0 or greater')
        pdbfile = 'PDB/4ZHL.cif'
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            model = self.cifparser.get_structure('4ZHL', pdbfile)[0]
            dssp = DSSP(model, pdbfile)
        self.assertEqual(len(dssp), 257)

    def test_dssp_with_mmcif_file_and_nonstandard_residues(self):
        if False:
            while True:
                i = 10
        'Test DSSP generation from MMCIF with non-standard residues.'
        if self.dssp_version < VERSION_2_2_0:
            self.skipTest('Test requires DSSP version 2.2.0 or greater')
        pdbfile = 'PDB/1AS5.cif'
        model = self.cifparser.get_structure('1AS5', pdbfile)[0]
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            dssp = DSSP(model, pdbfile)
        self.assertEqual(len(dssp), 24)

    def test_dssp_with_mmcif_file_and_different_chain_ids(self):
        if False:
            while True:
                i = 10
        'Test DSSP generation from MMCIF which has different label and author chain IDs.'
        if self.dssp_version < VERSION_2_2_0:
            self.skipTest('Test requires DSSP version 2.2.0 or greater')
        pdbfile = 'PDB/1A7G.cif'
        model = self.cifparser.get_structure('1A7G', pdbfile)[0]
        dssp = DSSP(model, pdbfile)
        self.assertEqual(len(dssp), 82)
        self.assertEqual(dssp.keys()[0][0], 'E')

class DSSP_test(unittest.TestCase):
    """Tests for DSSP parsing etc which don't need the binary tool."""

    def test_DSSP_file(self):
        if False:
            while True:
                i = 10
        'Test parsing of pregenerated DSSP.'
        (dssp, keys) = make_dssp_dict('PDB/2BEG.dssp')
        self.assertEqual(len(dssp), 130)

    def test_DSSP_noheader_file(self):
        if False:
            for i in range(10):
                print('nop')
        'Test parsing of pregenerated DSSP missing header information.'
        (dssp, keys) = make_dssp_dict('PDB/2BEG_noheader.dssp')
        self.assertEqual(len(dssp), 130)

    def test_DSSP_hbonds(self):
        if False:
            for i in range(10):
                print('nop')
        'Test parsing of DSSP hydrogen bond information.'
        (dssp, keys) = make_dssp_dict('PDB/2BEG.dssp')
        dssp_indices = {v[5] for v in dssp.values()}
        hb_indices = set()
        for val in dssp.values():
            hb_indices |= {val[5] + x for x in (val[6], val[8], val[10], val[12])}
        self.assertEqual(dssp_indices & hb_indices, hb_indices)

    def test_DSSP_in_model_obj(self):
        if False:
            for i in range(10):
                print('nop')
        'All elements correctly added to xtra attribute of input model object.'
        p = PDBParser()
        s = p.get_structure('example', 'PDB/2BEG.pdb')
        m = s[0]
        _ = DSSP(m, 'PDB/2BEG.dssp', 'dssp', 'Sander', 'DSSP')
        i = 0
        with open('PDB/dssp_xtra_Sander.txt') as fh_ref:
            ref_lines = fh_ref.readlines()
            for chain in m:
                for res in chain:
                    xtra_list_ref = ref_lines[i].rstrip().split('\t')
                    xtra_list_ref = list(map(will_it_float, xtra_list_ref))
                    xtra_itemts = sorted(res.xtra.items(), key=lambda s: s[0])
                    xtra_list = [t[1] for t in xtra_itemts]
                    xtra_list = list(map(will_it_float, xtra_list))
                    self.assertEqual(xtra_list, xtra_list_ref)
                    i += 1

    def test_DSSP_RSA(self):
        if False:
            print('Hello World!')
        'Tests the usage of different ASA tables.'
        p = PDBParser()
        s = p.get_structure('example', 'PDB/2BEG.pdb')
        m = s[0]
        _ = DSSP(m, 'PDB/2BEG.dssp', 'dssp', 'Sander', 'DSSP')
        i = 0
        with open('PDB/Sander_RASA.txt') as fh_ref:
            ref_lines = fh_ref.readlines()
            for chain in m:
                for res in chain:
                    rasa_ref = float(ref_lines[i].rstrip())
                    rasa = float(res.xtra['EXP_DSSP_RASA'])
                    self.assertAlmostEqual(rasa, rasa_ref)
                    i += 1
        s = p.get_structure('example', 'PDB/2BEG.pdb')
        m = s[0]
        _ = DSSP(m, 'PDB/2BEG.dssp', 'dssp', 'Wilke', 'DSSP')
        i = 0
        with open('PDB/Wilke_RASA.txt') as fh_ref:
            ref_lines = fh_ref.readlines()
            for chain in m:
                for res in chain:
                    rasa_ref = float(ref_lines[i].rstrip())
                    rasa = float(res.xtra['EXP_DSSP_RASA'])
                    self.assertAlmostEqual(rasa, rasa_ref)
                    i += 1
        s = p.get_structure('example', 'PDB/2BEG.pdb')
        m = s[0]
        _ = DSSP(m, 'PDB/2BEG.dssp', 'dssp', 'Miller', 'DSSP')
        i = 0
        with open('PDB/Miller_RASA.txt') as fh_ref:
            ref_lines = fh_ref.readlines()
            for chain in m:
                for res in chain:
                    rasa_ref = float(ref_lines[i].rstrip())
                    rasa = float(res.xtra['EXP_DSSP_RASA'])
                    self.assertAlmostEqual(rasa, rasa_ref)
                    i += 1
        s = p.get_structure('example', 'PDB/2BEG.pdb')
        m = s[0]
        _ = DSSP(m, 'PDB/2BEG.dssp', 'dssp', 'Ahmad', 'DSSP')
        i = 0
        with open('PDB/Ahmad_RASA.txt') as fh_ref:
            ref_lines = fh_ref.readlines()
            for chain in m:
                for res in chain:
                    rasa_ref = float(ref_lines[i].rstrip())
                    rasa = float(res.xtra['EXP_DSSP_RASA'])
                    self.assertAlmostEqual(rasa, rasa_ref)
                    i += 1
if __name__ == '__main__':
    runner = unittest.TextTestRunner(verbosity=2)
    unittest.main(testRunner=runner)