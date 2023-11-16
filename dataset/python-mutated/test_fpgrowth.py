import unittest
import numpy as np
from test_fpbase import FPTestEdgeCases, FPTestErrors, FPTestEx1All, FPTestEx2All, FPTestEx3All
from mlxtend.frequent_patterns import fpgrowth

class TestEdgeCases(unittest.TestCase, FPTestEdgeCases):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        FPTestEdgeCases.setUp(self, fpgrowth)

class TestErrors(unittest.TestCase, FPTestErrors):

    def setUp(self):
        if False:
            return 10
        FPTestErrors.setUp(self, fpgrowth)

class TestEx1(unittest.TestCase, FPTestEx1All):

    def setUp(self):
        if False:
            while True:
                i = 10
        FPTestEx1All.setUp(self, fpgrowth)

class TestEx1BoolInput(unittest.TestCase, FPTestEx1All):

    def setUp(self):
        if False:
            return 10
        one_ary = np.array([[False, False, False, True, False, True, True, True, True, False, True], [False, False, True, True, False, True, False, True, True, False, True], [True, False, False, True, False, True, True, False, False, False, False], [False, True, False, False, False, True, True, False, False, True, True], [False, True, False, True, True, True, False, False, True, False, False]])
        FPTestEx1All.setUp(self, fpgrowth, one_ary=one_ary)

class TestEx2(unittest.TestCase, FPTestEx2All):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        FPTestEx2All.setUp(self, fpgrowth)

class TestEx3(unittest.TestCase, FPTestEx3All):

    def setUp(self):
        if False:
            print('Hello World!')
        FPTestEx3All.setUp(self, fpgrowth)