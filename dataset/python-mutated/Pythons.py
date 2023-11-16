""" Test tool to run a program with various Pythons. """
from nuitka.PythonVersions import getSupportedPythonVersions
from nuitka.utils.Execution import check_output
from nuitka.utils.InstalledPythons import findPythons

def findAllPythons():
    if False:
        return 10
    for python_version in getSupportedPythonVersions():
        for python in findPythons(python_version):
            yield (python, python_version)

def executeWithInstalledPython(python, args):
    if False:
        for i in range(10):
            print('nop')
    return check_output([python.getPythonExe()] + args)