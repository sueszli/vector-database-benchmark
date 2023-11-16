import os
import unittest
import py_compile
import TestTools

class Tests(TestTools.BaseToolsTest):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        TestTools.BaseToolsTest.setUp(self)

    def SingleFileTest(self, filename):
        if False:
            for i in range(10):
                print('nop')
        try:
            py_compile.compile(filename, doraise=True)
        except Exception as e:
            self.fail('syntax error: %s, Error is %s' % (filename, str(e)))

def MakePythonSyntaxCheckTests():
    if False:
        for i in range(10):
            print('nop')

    def GetAllPythonSourceFiles():
        if False:
            while True:
                i = 10
        pythonSourceFiles = []
        for (root, dirs, files) in os.walk(TestTools.PythonSourceDir):
            for filename in files:
                if filename.lower().endswith('.py'):
                    pythonSourceFiles.append(os.path.join(root, filename))
        return pythonSourceFiles

    def MakeTestName(filename):
        if False:
            print('Hello World!')
        assert filename.lower().endswith('.py')
        name = filename[:-3]
        name = name.replace(TestTools.PythonSourceDir, '')
        name = name.replace(os.path.sep, '_')
        return 'test' + name

    def MakeNewTest(filename):
        if False:
            while True:
                i = 10
        test = MakeTestName(filename)
        newmethod = lambda self: self.SingleFileTest(filename)
        setattr(Tests, test, newmethod)
    for filename in GetAllPythonSourceFiles():
        MakeNewTest(filename)
MakePythonSyntaxCheckTests()
del MakePythonSyntaxCheckTests
TheTestSuite = TestTools.MakeTheTestSuite(locals())
if __name__ == '__main__':
    allTests = TheTestSuite()
    unittest.TextTestRunner().run(allTests)