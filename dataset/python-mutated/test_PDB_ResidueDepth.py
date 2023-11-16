"""Unit tests for the Bio.PDB.ResidueDepth module."""
import subprocess
import unittest
import warnings
from Bio.PDB import MMCIFParser, PDBParser, ResidueDepth
from Bio.PDB.PDBExceptions import PDBConstructionWarning
from Bio.PDB.ResidueDepth import _get_atom_radius

class MSMS_tests(unittest.TestCase):
    """Test calling MSMS via Bio.PDB.ResidueDepth."""

    @classmethod
    def setUpClass(cls):
        if False:
            i = 10
            return i + 15
        try:
            v = subprocess.check_output(['msms', '-h'], text=True, stderr=subprocess.STDOUT)
        except OSError:
            raise unittest.SkipTest('Install MSMS if you want to use it from Biopython.')
        cls.pdbparser = PDBParser()
        cls.cifparser = MMCIFParser()

    def check_msms(self, prot_file, first_100_residues):
        if False:
            print('Hello World!')
        'Wrap calls to MSMS and the respective tests.'
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', PDBConstructionWarning)
            s = self.pdbparser.get_structure('X', prot_file)
        model = s[0]
        rd = ResidueDepth(model)
        residues = []
        for item in rd.property_list[:100]:
            residues.append(item[0].get_resname())
        self.assertEqual(''.join(residues), first_100_residues)

    def test_ResidueDepth_2BEG(self):
        if False:
            print('Hello World!')
        self.check_msms('PDB/2BEG.pdb', 'LEUVALPHEPHEALAGLUASPVALGLYSERASNLYSGLYALAILEILEGLYLEUMETVALGLYGLYVALVALILEALALEUVALPHEPHEALAGLUASPVALGLYSERASNLYSGLYALAILEILEGLYLEUMETVALGLYGLYVALVALILEALALEUVALPHEPHEALAGLUASPVALGLYSERASNLYSGLYALAILEILEGLYLEUMETVALGLYGLYVALVALILEALALEUVALPHEPHEALAGLUASPVALGLYSERASNLYSGLYALAILEILEGLYLEUMETVALGLYGLY')

    def test_ResidueDepth_1LCD(self):
        if False:
            i = 10
            return i + 15
        self.check_msms('PDB/1LCD.pdb', 'METLYSPROVALTHRLEUTYRASPVALALAGLUTYRALAGLYVALSERTYRGLNTHRVALSERARGVALVALASNGLNALASERHISVALSERALALYSTHRARGGLULYSVALGLUALAALAMETALAGLULEUASNTYRILEPROASNARG')

    def test_ResidueDepth_1A8O(self):
        if False:
            for i in range(10):
                print('nop')
        self.check_msms('PDB/1A8O.pdb', 'MSEASPILEARGGLNGLYPROLYSGLUPROPHEARGASPTYRVALASPARGPHETYRLYSTHRLEUARGALAGLUGLNALASERGLNGLUVALLYSASNTRPMSETHRGLUTHRLEULEUVALGLNASNALAASNPROASPCYSLYSTHRILELEULYSALALEUGLYPROGLYALATHRLEUGLUGLUMSEMSETHRALACYSGLNGLY')

class ResidueDepth_tests(unittest.TestCase):
    """Tests for Bio.PDB.ResidueDepth, except for running MSMS itself."""

    def test_pdb_to_xyzr(self):
        if False:
            return 10
        'Test generation of xyzr (atomic radii) file.'
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', PDBConstructionWarning)
            p = PDBParser(PERMISSIVE=1)
            structure = p.get_structure('example', 'PDB/1A8O.pdb')
        with open('PDB/1A8O.xyzr') as handle:
            msms_radii = []
            for line in handle:
                fields = line.split()
                radius = float(fields[3])
                msms_radii.append(radius)
        model = structure[0]
        biopy_radii = []
        for atom in model.get_atoms():
            biopy_radii.append(_get_atom_radius(atom, rtype='united'))
        self.assertEqual(msms_radii, biopy_radii)
if __name__ == '__main__':
    runner = unittest.TextTestRunner(verbosity=2)
    unittest.main(testRunner=runner)