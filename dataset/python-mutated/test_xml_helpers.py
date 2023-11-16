"""Module containing tests for xml_helpers Module"""
import os
import sys
import unittest
import six
sys.path.append('.')
from pywinauto.xml_helpers import WriteDialogToFile
from pywinauto.xml_helpers import ReadPropertiesFromFile
from pywinauto.xml_helpers import LOGFONTW
from pywinauto.xml_helpers import RECT

class XMLHelperTestCases(unittest.TestCase):
    """Unit tests for the ListViewWrapper class"""

    def tearDown(self):
        if False:
            print('Hello World!')
        'delete the file we have created'
        os.unlink('__unittests.xml')

    def assertReadWriteSame(self, props):
        if False:
            i = 10
            return i + 15
        'Make sure that roundtripping produces identical file'
        WriteDialogToFile('__unittests.xml', props)
        read_props = ReadPropertiesFromFile('__unittests.xml')
        self.assertEqual(props, read_props)

    def testOneUnicode(self):
        if False:
            i = 10
            return i + 15
        'Test writing/reading a unicode string'
        props = [dict(test=u'hiya')]
        self.assertReadWriteSame(props)

    def testOneString(self):
        if False:
            return 10
        'Test writing/reading a string'
        props = [dict(test='hiya')]
        self.assertReadWriteSame(props)

    def testSomeEscapes(self):
        if False:
            return 10
        'Test writing/reading a dictionary with some escape sequences'
        test_string = []
        for i in range(0, 50000):
            test_string.append(six.unichr(i))
        test_string = ''.join(test_string)
        props = [dict(test=test_string)]
        self.assertReadWriteSame(props)

    def testOneBool(self):
        if False:
            while True:
                i = 10
        'Test writing/reading Bool'
        props = [dict(test=True)]
        self.assertReadWriteSame(props)

    def testOneList(self):
        if False:
            print('Hello World!')
        'Test writing/reading a list'
        props = [dict(test=[1, 2, 3, 4, 5, 6])]
        self.assertReadWriteSame(props)

    def testOneDict(self):
        if False:
            return 10
        'Test writing/reading a dictionary with one element'
        props = [dict(test_value=dict(test=1))]
        self.assertReadWriteSame(props)

    def testOneLong(self):
        if False:
            i = 10
            return i + 15
        'Test writing/reading one long is correct'
        props = [dict(test=1)]
        self.assertReadWriteSame(props)

    def testLOGFONTW(self):
        if False:
            while True:
                i = 10
        'Test writing/reading one LOGFONTW is correct'
        font = LOGFONTW()
        font.lfWeight = 23
        font.lfFaceName = u'wowow'
        props = [dict(test=font)]
        self.assertReadWriteSame(props)

    def testRECT(self):
        if False:
            while True:
                i = 10
        'Test writing/reading one RECT is correct'
        props = [dict(test=RECT(1, 2, 3, 4))]
        self.assertReadWriteSame(props)

    def testTwoLong(self):
        if False:
            for i in range(10):
                print('nop')
        'Test writing/reading two longs is correct'
        props = [dict(test=1), dict(test_blah=2)]
        self.assertReadWriteSame(props)

    def testEmptyList(self):
        if False:
            print('Hello World!')
        'Test writing/reading empty list'
        props = [dict(test=[])]
        self.assertReadWriteSame(props)

    def testEmptyDict(self):
        if False:
            return 10
        'Test writing/reading empty dict'
        props = [dict(test={})]
        self.assertReadWriteSame(props)
if __name__ == '__main__':
    unittest.main()