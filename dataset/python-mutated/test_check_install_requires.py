import unittest
from pocsuite3.lib.core.register import PocLoader

class TestCase(unittest.TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        pass

    def test_module_is_none(self):
        if False:
            i = 10
            return i + 15
        p = PocLoader('testcase', 'testcase')
        p.set_data('install_requires = [\'\', ""]')
        p.check_requires(p.data)

    def test_built_in_module(self):
        if False:
            while True:
                i = 10
        p = PocLoader('testcase', 'testcase')
        p.set_data("install_requires = ['os', 'sys']")
        p.check_requires(p.data)

    def test_normal_module(self):
        if False:
            print('Hello World!')
        p = PocLoader('testcase', 'testcase')
        p.set_data("install_requires = ['setuptools']")
        p.check_requires(p.data)

    def test_module_include_version(self):
        if False:
            i = 10
            return i + 15
        p = PocLoader('testcase', 'testcase')
        p.set_data("install_requires = ['setuptools==51.1.2']")
        p.check_requires(p.data)
        p.set_data("install_requires = ['setuptools~=51.1.2']")
        p.check_requires(p.data)
        p.set_data("install_requires = ['setuptools>=51.1.2']")
        p.check_requires(p.data)
        p.set_data("install_requires = ['setuptools<=51.1.2']")
        p.check_requires(p.data)

    def test_import_name_and_install_name_are_inconsistent(self):
        if False:
            i = 10
            return i + 15
        p = PocLoader('testcase', 'testcase')
        p.set_data("install_requires = ['BeautifulSoup4>=4.9.1:bs4']")
        try:
            p.check_requires(p.data)
        except SystemExit:
            pass
if __name__ == '__main__':
    unittest.main()