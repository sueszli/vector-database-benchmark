from __future__ import print_function
import unittest
from binascii import hexlify
from impacket.dcerpc.v5.ndr import NDRSTRUCT, NDRLONG, NDRSHORT, NDRUniFixedArray, NDRUniVaryingArray, NDRUniConformantVaryingArray, NDRVaryingString, NDRConformantVaryingString, NDRPOINTERNULL

def hexl(b):
    if False:
        print('Hello World!')
    hexstr = str(hexlify(b).decode('ascii'))
    return ' '.join([hexstr[i:i + 8] for i in range(0, len(hexstr), 8)])

class NDRTest(object):

    def create(self, data=None, isNDR64=False):
        if False:
            for i in range(10):
                print('nop')
        if data is not None:
            return self.theClass(data, isNDR64=isNDR64)
        else:
            return self.theClass(isNDR64=isNDR64)

    def do_test(self, isNDR64=False):
        if False:
            while True:
                i = 10
        a = self.create(isNDR64=isNDR64)
        self.populate(a)
        a_str = a.getData()
        self.check_data(a_str, isNDR64)
        b = self.create(a_str, isNDR64=isNDR64)
        b_str = b.getData()
        self.assertEqual(b_str, a_str)

    def test_false(self):
        if False:
            return 10
        self.do_test(False)

    def test_true(self):
        if False:
            while True:
                i = 10
        self.do_test(True)

    def check_data(self, a_str, isNDR64):
        if False:
            i = 10
            return i + 15
        try:
            hexData = getattr(self, 'hexData64' if isNDR64 else 'hexData')
            self.assertEqual(hexl(a_str), hexData)
        except AttributeError:
            print(self.__class__.__name__, isNDR64, hexl(a_str))

class TestUniFixedArray(NDRTest, unittest.TestCase):

    class theClass(NDRSTRUCT):
        structure = (('Array', NDRUniFixedArray),)

    def populate(self, a):
        if False:
            for i in range(10):
                print('nop')
        a['Array'] = b'12345678'
    hexData = '31323334 35363738'
    hexData64 = hexData

class TestStructWithPad(NDRTest, unittest.TestCase):

    class theClass(NDRSTRUCT):
        structure = (('long', NDRLONG), ('short', NDRSHORT))

    def populate(self, a):
        if False:
            for i in range(10):
                print('nop')
        a['long'] = 170
        a['short'] = 187
    hexData = 'aa000000 bb00'
    hexData64 = hexData

class TestUniVaryingArray(NDRTest, unittest.TestCase):

    class theClass(NDRSTRUCT):
        structure = (('Array', NDRUniVaryingArray),)

    def populate(self, a):
        if False:
            return 10
        a['Array'] = b'12345678'
    hexData = '00000000 08000000 31323334 35363738'
    hexData64 = '00000000 00000000 08000000 00000000 31323334 35363738'

class TestUniConformantVaryingArray(NDRTest, unittest.TestCase):

    class theClass(NDRSTRUCT):
        structure = (('Array', NDRUniConformantVaryingArray),)

    def populate(self, a):
        if False:
            for i in range(10):
                print('nop')
        a['Array'] = b'12345678'
    hexData = '08000000 00000000 08000000 31323334 35363738'
    hexData64 = '08000000 00000000 00000000 00000000 08000000 00000000 31323334 35363738'

class TestVaryingString(NDRTest, unittest.TestCase):

    class theClass(NDRSTRUCT):
        structure = (('Array', NDRVaryingString),)

    def populate(self, a):
        if False:
            while True:
                i = 10
        a['Array'] = b'12345678'
    hexData = '00000000 09000000 31323334 35363738 00'
    hexData64 = '00000000 00000000 09000000 00000000 31323334 35363738 00'

class TestConformantVaryingString(NDRTest, unittest.TestCase):

    class theClass(NDRSTRUCT):
        structure = (('Array', NDRConformantVaryingString),)

    def populate(self, a):
        if False:
            print('Hello World!')
        a['Array'] = b'12345678'
    hexData = '08000000 00000000 08000000 31323334 35363738'
    hexData64 = '08000000 00000000 00000000 00000000 08000000 00000000 31323334 35363738'

class TestPointerNULL(NDRTest, unittest.TestCase):

    class theClass(NDRSTRUCT):
        structure = (('Array', NDRPOINTERNULL),)

    def populate(self, a):
        if False:
            while True:
                i = 10
        pass
    hexData = '00000000'
    hexData64 = '00000000 00000000'
if __name__ == '__main__':
    unittest.main(verbosity=1)