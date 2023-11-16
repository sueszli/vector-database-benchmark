from __future__ import print_function
import os
import sys
import unittest
import coverage
testfolder = os.path.abspath(os.path.dirname(__file__))
package_root = os.path.abspath(os.path.join(testfolder, '..\\..'))
sys.path.append(package_root)
cov = coverage.coverage(branch=True, omit=os.path.join(package_root, 'pywinauto', '*tests', '*.py'))
cov.start()
import pywinauto
pywinauto.actionlogger.enable()
modules_to_test = [pywinauto]

def run_tests():
    if False:
        for i in range(10):
            print('nop')
    excludes = []
    suite = unittest.TestSuite()
    sys.path.append(testfolder)
    for (root, dirs, files) in os.walk(testfolder):
        test_modules = [file.replace('.py', '') for file in files if file.startswith('test_') and file.endswith('.py')]
        test_modules = [mod for mod in test_modules if mod.lower() not in excludes]
        for mod in test_modules:
            imported_mod = __import__(mod, globals(), locals())
            suite.addTests(unittest.defaultTestLoader.loadTestsFromModule(imported_mod))
    unittest.TextTestRunner(verbosity=1).run(suite)
    cov.stop()
    print(cov.report())
    cov.html_report(directory=os.path.join(package_root, 'Coverage_report'), omit=[os.path.join(package_root, 'pywinauto', '*tests', '*.py'), os.path.join(package_root, 'pywinauto', 'six.py')])
if __name__ == '__main__':
    run_tests()