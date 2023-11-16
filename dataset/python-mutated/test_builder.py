import unittest
from patterns.creational.builder import ComplexHouse, Flat, House, construct_building

class TestSimple(unittest.TestCase):

    def test_house(self):
        if False:
            while True:
                i = 10
        house = House()
        self.assertEqual(house.size, 'Big')
        self.assertEqual(house.floor, 'One')

    def test_flat(self):
        if False:
            for i in range(10):
                print('nop')
        flat = Flat()
        self.assertEqual(flat.size, 'Small')
        self.assertEqual(flat.floor, 'More than One')

class TestComplex(unittest.TestCase):

    def test_house(self):
        if False:
            for i in range(10):
                print('nop')
        house = construct_building(ComplexHouse)
        self.assertEqual(house.size, 'Big and fancy')
        self.assertEqual(house.floor, 'One')