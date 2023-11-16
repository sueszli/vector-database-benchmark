import os
import sys
import unittest
import TianoCompress
modules = (TianoCompress,)

def TheTestSuite():
    if False:
        print('Hello World!')
    suites = list(map(lambda module: module.TheTestSuite(), modules))
    return unittest.TestSuite(suites)
if __name__ == '__main__':
    allTests = TheTestSuite()
    unittest.TextTestRunner().run(allTests)