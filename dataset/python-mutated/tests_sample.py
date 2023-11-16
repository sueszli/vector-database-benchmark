import doctest
from unittest import TestCase
from django.test import SimpleTestCase
from django.test import TestCase as DjangoTestCase
from . import doctests

class TestVanillaUnittest(TestCase):

    def test_sample(self):
        if False:
            while True:
                i = 10
        self.assertEqual(1, 1)

class TestDjangoTestCase(DjangoTestCase):

    def test_sample(self):
        if False:
            print('Hello World!')
        self.assertEqual(1, 1)

class TestZimpleTestCase(SimpleTestCase):

    def test_sample(self):
        if False:
            print('Hello World!')
        self.assertEqual(1, 1)

class EmptyTestCase(TestCase):
    pass

def load_tests(loader, tests, ignore):
    if False:
        print('Hello World!')
    tests.addTests(doctest.DocTestSuite(doctests))
    return tests