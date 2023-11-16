import errno
import pytest
import subprocess
from os import path

def test_version_print():
    if False:
        i = 10
        return i + 15
    from wal_e import cmd
    place = path.join(path.dirname(cmd.__file__), 'VERSION')
    with open(place, 'rb') as f:
        expected = f.read()
    try:
        proc = subprocess.Popen(['wal-e', 'version'], stdout=subprocess.PIPE)
    except EnvironmentError as e:
        if e.errno == errno.ENOENT:
            pytest.skip('wal-e must be in $PATH to test version output')
    result = proc.communicate()[0]
    assert proc.returncode == 0
    assert result == expected