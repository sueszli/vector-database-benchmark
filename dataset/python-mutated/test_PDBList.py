"""Testing access to the PDB over the internet."""
import contextlib
import os
import shutil
import tempfile
import unittest
from Bio.PDB.PDBList import PDBList
import requires_internet
requires_internet.check()

class TestPBDListGetList(unittest.TestCase):
    """Test methods responsible for getting lists of entries."""

    def test_get_recent_changes(self):
        if False:
            return 10
        'Tests the Bio.PDB.PDBList.get_recent_changes method.'
        pdblist = PDBList(obsolete_pdb='unimportant')
        url = pdblist.pdb_server + '/pub/pdb/data/status/latest/added.pdb'
        entries = pdblist.get_status_list(url)
        self.assertIsNotNone(entries)

    def test_get_all_entries(self):
        if False:
            print('Hello World!')
        'Tests the Bio.PDB.PDBList.get_all_entries method.'
        pdblist = PDBList(obsolete_pdb='unimportant')
        entries = pdblist.get_all_entries()
        self.assertGreater(len(entries), 100000)

    def test_get_all_obsolete(self):
        if False:
            return 10
        'Tests the Bio.PDB.PDBList.get_all_obsolete method.'
        pdblist = PDBList(obsolete_pdb='unimportant')
        entries = pdblist.get_all_obsolete()
        self.assertGreater(len(entries), 3000)

    def test_get_all_assemblies(self):
        if False:
            for i in range(10):
                print('nop')
        'Tests the Bio.PDB.PDBList.get_all_assemblies method.'
        pdblist = PDBList(obsolete_pdb='unimportant')
        entries = pdblist.get_all_assemblies()
        self.assertGreater(len(entries), 100000)

class TestPDBListGetStructure(unittest.TestCase):
    """Test methods responsible for getting structures."""

    @contextlib.contextmanager
    def make_temp_directory(self, directory):
        if False:
            i = 10
            return i + 15
        temp_dir = tempfile.mkdtemp(dir=directory)
        try:
            yield temp_dir
        finally:
            shutil.rmtree(temp_dir)

    def check(self, structure, filename, file_format, obsolete=False, pdir=None):
        if False:
            while True:
                i = 10
        with self.make_temp_directory(os.getcwd()) as tmp:
            pdblist = PDBList(pdb=tmp)
            path = os.path.join(tmp, filename)
            if pdir:
                pdir = os.path.join(tmp, pdir)
            pdblist.retrieve_pdb_file(structure, obsolete=obsolete, pdir=pdir, file_format=file_format)
            self.assertTrue(os.path.isfile(path))

    def test_retrieve_pdb_file_small_pdb(self):
        if False:
            for i in range(10):
                print('nop')
        'Tests retrieving the small molecule in pdb format.'
        structure = '127d'
        self.check(structure, os.path.join(structure[1:3], f'pdb{structure}.ent'), 'pdb')

    def test_retrieve_pdb_file_large_pdb(self):
        if False:
            print('Hello World!')
        'Tests retrieving the bundle for large molecule in pdb-like format.'
        structure = '3k1q'
        self.check(structure, os.path.join(structure[1:3], f'{structure}-pdb-bundle.tar'), 'bundle')

    def test_retrieve_pdb_file_obsolete_pdb(self):
        if False:
            while True:
                i = 10
        'Tests retrieving the obsolete molecule in pdb format.'
        structure = '347d'
        self.check(structure, os.path.join('obsolete', structure[1:3], f'pdb{structure}.ent'), 'pdb', obsolete=True)

    def test_retrieve_pdb_file_obsolete_mmcif(self):
        if False:
            i = 10
            return i + 15
        'Tests retrieving the obsolete molecule in mmcif format.'
        structure = '347d'
        self.check(structure, os.path.join('obsolete', structure[1:3], f'{structure}.cif'), 'mmCif', obsolete=True)

    def test_retrieve_pdb_file_mmcif(self):
        if False:
            i = 10
            return i + 15
        'Tests retrieving the (non-obsolete) molecule in mmcif format.'
        structure = '127d'
        self.check(structure, os.path.join(structure[1:3], f'{structure}.cif'), 'mmCif')

    def test_retrieve_pdb_file_obsolete_xml(self):
        if False:
            while True:
                i = 10
        'Tests retrieving the obsolete molecule in mmcif format.'
        structure = '347d'
        self.check(structure, os.path.join('obsolete', structure[1:3], f'{structure}.xml'), 'xml', obsolete=True)

    def test_retrieve_pdb_file_xml(self):
        if False:
            print('Hello World!')
        'Tests retrieving the (non obsolete) molecule in xml format.'
        structure = '127d'
        self.check(structure, os.path.join(structure[1:3], f'{structure}.xml'), 'xml')

    def test_retrieve_pdb_file_mmtf(self):
        if False:
            return 10
        'Tests retrieving the molecule in mmtf format.'
        structure = '127d'
        self.check(structure, os.path.join(structure[1:3], f'{structure}.mmtf'), 'mmtf')

    def test_double_retrieve_structure(self):
        if False:
            for i in range(10):
                print('nop')
        'Tests retrieving the same file to different directories.'
        structure = '127d'
        self.check(structure, os.path.join('a', f'{structure}.cif'), 'mmCif', pdir='a')
        self.check(structure, os.path.join('b', f'{structure}.cif'), 'mmCif', pdir='b')

class TestPDBListGetAssembly(unittest.TestCase):
    """Test methods responsible for getting assemblies."""

    @contextlib.contextmanager
    def make_temp_directory(self, directory):
        if False:
            for i in range(10):
                print('nop')
        temp_dir = tempfile.mkdtemp(dir=directory)
        try:
            yield temp_dir
        finally:
            shutil.rmtree(temp_dir)

    def check(self, structure, assembly_num, filename, file_format, pdir=None):
        if False:
            for i in range(10):
                print('nop')
        with self.make_temp_directory(os.getcwd()) as tmp:
            pdblist = PDBList(pdb=tmp)
            path = os.path.join(tmp, filename)
            if pdir:
                pdir = os.path.join(tmp, pdir)
            pdblist.retrieve_assembly_file(structure, assembly_num, pdir=pdir, file_format=file_format)
            self.assertTrue(os.path.isfile(path))

    def test_retrieve_assembly_file_mmcif(self):
        if False:
            i = 10
            return i + 15
        'Tests retrieving a small assembly in mmCif format.'
        structure = '127d'
        assembly_num = '1'
        self.check(structure, assembly_num, os.path.join(structure[1:3], f'{structure}-assembly{assembly_num}.cif'), 'mmCif')

    def test_retrieve_assembly_file_pdb(self):
        if False:
            i = 10
            return i + 15
        'Tests retrieving a small assembly in pdb format.'
        structure = '127d'
        assembly_num = '1'
        self.check(structure, assembly_num, os.path.join(structure[1:3], f'{structure}.pdb{assembly_num}'), 'pdb')

    def test_double_retrieve_assembly(self):
        if False:
            return 10
        'Tests retrieving the same file to different directories.'
        structure = '127d'
        assembly_num = '1'
        self.check(structure, assembly_num, os.path.join('a', f'{structure}-assembly{assembly_num}.cif'), 'mmCif', pdir='a')
        self.check(structure, assembly_num, os.path.join('b', f'{structure}-assembly{assembly_num}.cif'), 'mmCif', pdir='b')
if __name__ == '__main__':
    runner = unittest.TextTestRunner(verbosity=2)
    unittest.main(testRunner=runner)