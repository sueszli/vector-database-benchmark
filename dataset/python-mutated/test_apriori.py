import unittest
import numpy as np
from test_fpbase import FPTestEdgeCases, FPTestErrors, FPTestEx1All, FPTestEx2All, FPTestEx3All
from mlxtend.frequent_patterns import apriori

def apriori_wrapper_low_memory(*args, **kwargs):
    if False:
        print('Hello World!')
    return apriori(*args, **kwargs, low_memory=True)

class TestEdgeCases(unittest.TestCase, FPTestEdgeCases):

    def setUp(self):
        if False:
            while True:
                i = 10
        FPTestEdgeCases.setUp(self, apriori)

class TestErrors(unittest.TestCase, FPTestErrors):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        FPTestErrors.setUp(self, apriori)

class TestApriori(unittest.TestCase, FPTestEx1All):

    def setUp(self):
        if False:
            print('Hello World!')
        FPTestEx1All.setUp(self, apriori)

class TestAprioriLowMemory(unittest.TestCase, FPTestEx1All):

    def setUp(self):
        if False:
            return 10
        FPTestEx1All.setUp(self, apriori_wrapper_low_memory)

class TestAprioriBoolInput(unittest.TestCase, FPTestEx1All):

    def setUp(self):
        if False:
            return 10
        one_ary = np.array([[False, False, False, True, False, True, True, True, True, False, True], [False, False, True, True, False, True, False, True, True, False, True], [True, False, False, True, False, True, True, False, False, False, False], [False, True, False, False, False, True, True, False, False, True, True], [False, True, False, True, True, True, False, False, True, False, False]])
        FPTestEx1All.setUp(self, apriori, one_ary=one_ary)

class TestEx2(unittest.TestCase, FPTestEx2All):

    def setUp(self):
        if False:
            while True:
                i = 10
        FPTestEx2All.setUp(self, apriori)

class TestEx3(unittest.TestCase, FPTestEx3All):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        FPTestEx3All.setUp(self, apriori)