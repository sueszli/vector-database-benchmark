import copy
import os
import subprocess
import xml.etree.ElementTree as ET
import xml.etree.cElementTree as cET
from pyi_testmod_gettemp import gettemp
_pyexe_file = gettemp('python_exe.build')
with open(_pyexe_file) as fp:
    _lines = fp.readlines()
    _pyexe = _lines[0].strip()
    _env_path = _lines[2].strip()

def exec_python(pycode):
    if False:
        while True:
            i = 10
    '\n    Wrap running python script in a subprocess.\n\n    Return stdout of the invoked command.\n    '
    env = copy.deepcopy(os.environ)
    env['PATH'] = _env_path
    out = subprocess.Popen([_pyexe, '-c', pycode], env=env, stdout=subprocess.PIPE, shell=False).stdout.read()
    out = out.decode('ascii').strip()
    return out

def compare(test_name, expect, frozen):
    if False:
        i = 10
        return i + 15
    if '__cached__' not in frozen:
        frozen.append('__cached__')
    frozen.sort()
    frozen = str(frozen)
    print(test_name)
    print('  Attributes expected: ' + expect)
    print('  Attributes current:  ' + frozen)
    print('')
    if not frozen == expect:
        raise SystemExit('Frozen module has no same attributes as unfrozen.')
subproc_code = '\nimport {0} as myobject\nlst = dir(myobject)\n# Sort attributes.\nlst.sort()\nprint(lst)\n'
_expect = exec_python(subproc_code.format('xml.etree.ElementTree'))
_frozen = dir(ET)
compare('ElementTree', _expect, _frozen)
_expect = exec_python(subproc_code.format('xml.etree.cElementTree'))
_frozen = dir(cET)
compare('cElementTree', _expect, _frozen)