from unittest import TestCase
from golem.core.printable_object import PrintableObject

class PrintableObjectSpecimen(PrintableObject):

    def __init__(self, one, two):
        if False:
            print('Hello World!')
        self.one = one
        self.two = two

class PrintableObjectTest(TestCase):

    def test_printable_object(self):
        if False:
            i = 10
            return i + 15
        po = PrintableObjectSpecimen('phobos', 'deimos')
        self.assertEqual(str(po), 'PrintableObjectSpecimen <one: phobos, two: deimos>')