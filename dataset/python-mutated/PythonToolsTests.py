import os
import sys
import unittest

def TheTestSuite():
    if False:
        i = 10
        return i + 15
    suites = []
    import CheckPythonSyntax
    suites.append(CheckPythonSyntax.TheTestSuite())
    import CheckUnicodeSourceFiles
    suites.append(CheckUnicodeSourceFiles.TheTestSuite())
    return unittest.TestSuite(suites)
if __name__ == '__main__':
    allTests = TheTestSuite()
    unittest.TextTestRunner().run(allTests)