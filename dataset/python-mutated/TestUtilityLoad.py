import unittest
from Cython.Compiler import Code, UtilityCode

def strip_2tup(tup):
    if False:
        return 10
    return (tup[0] and tup[0].strip(), tup[1] and tup[1].strip())

class TestUtilityLoader(unittest.TestCase):
    """
    Test loading UtilityCodes
    """
    expected = ('test {{loader}} prototype', 'test {{loader}} impl')
    required = ('req {{loader}} proto', 'req {{loader}} impl')
    context = dict(loader='Loader')
    name = 'TestUtilityLoader'
    filename = 'TestUtilityLoader.c'
    cls = Code.UtilityCode

    def test_load_as_string(self):
        if False:
            return 10
        got = strip_2tup(self.cls.load_as_string(self.name, self.filename))
        self.assertEqual(got, self.expected)

    def test_load(self):
        if False:
            for i in range(10):
                print('nop')
        utility = self.cls.load(self.name, from_file=self.filename)
        got = strip_2tup((utility.proto, utility.impl))
        self.assertEqual(got, self.expected)
        (required,) = utility.requires
        got = strip_2tup((required.proto, required.impl))
        self.assertEqual(got, self.required)
        utility = self.cls.load_cached(self.name, from_file=self.filename)
        got = strip_2tup((utility.proto, utility.impl))
        self.assertEqual(got, self.expected)

class TestTempitaUtilityLoader(TestUtilityLoader):
    """
    Test loading UtilityCodes with Tempita substitution
    """
    expected_tempita = (TestUtilityLoader.expected[0].replace('{{loader}}', 'Loader'), TestUtilityLoader.expected[1].replace('{{loader}}', 'Loader'))
    required_tempita = (TestUtilityLoader.required[0].replace('{{loader}}', 'Loader'), TestUtilityLoader.required[1].replace('{{loader}}', 'Loader'))
    cls = Code.TempitaUtilityCode

    def test_load_as_string(self):
        if False:
            i = 10
            return i + 15
        got = strip_2tup(self.cls.load_as_string(self.name, self.filename, context=self.context))
        self.assertEqual(got, self.expected_tempita)

    def test_load(self):
        if False:
            i = 10
            return i + 15
        utility = self.cls.load(self.name, self.filename, context=self.context)
        got = strip_2tup((utility.proto, utility.impl))
        self.assertEqual(got, self.expected_tempita)
        (required,) = utility.requires
        got = strip_2tup((required.proto, required.impl))
        self.assertEqual(got, self.required_tempita)
        utility = self.cls.load(self.name, from_file=self.filename, context=self.context)
        got = strip_2tup((utility.proto, utility.impl))
        self.assertEqual(got, self.expected_tempita)

class TestCythonUtilityLoader(TestTempitaUtilityLoader):
    """
    Test loading CythonUtilityCodes
    """
    expected = (None, 'test {{cy_loader}} impl')
    expected_tempita = (None, 'test CyLoader impl')
    required = (None, 'req {{cy_loader}} impl')
    required_tempita = (None, 'req CyLoader impl')
    context = dict(cy_loader='CyLoader')
    name = 'TestCyUtilityLoader'
    filename = 'TestCyUtilityLoader.pyx'
    cls = UtilityCode.CythonUtilityCode
    cls.proto = None
    test_load = TestUtilityLoader.test_load
    test_load_tempita = TestTempitaUtilityLoader.test_load