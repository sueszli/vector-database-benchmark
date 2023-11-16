import os
import sys
import unittest
import TestTools

def GetCTestSuite():
    if False:
        print('Hello World!')
    import CToolsTests
    return CToolsTests.TheTestSuite()

def GetPythonTestSuite():
    if False:
        while True:
            i = 10
    import PythonToolsTests
    return PythonToolsTests.TheTestSuite()

def GetAllTestsSuite():
    if False:
        print('Hello World!')
    return unittest.TestSuite([GetCTestSuite(), GetPythonTestSuite()])
if __name__ == '__main__':
    allTests = GetAllTestsSuite()
    unittest.TextTestRunner(verbosity=2).run(allTests)