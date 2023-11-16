"""
Test that CVEs stay fixed. 
"""
from IPython.utils.tempdir import TemporaryDirectory, TemporaryWorkingDirectory
from pathlib import Path
import random
import sys
import os
import string
import subprocess

def test_cve_2022_21699():
    if False:
        print('Hello World!')
    '\n    Here we test CVE-2022-21699.\n\n    We create a temporary directory, cd into it.\n    Make a profile file that should not be executed and start IPython in a subprocess,\n    checking for the value.\n\n\n\n    '
    dangerous_profile_dir = Path('profile_default')
    dangerous_startup_dir = dangerous_profile_dir / 'startup'
    dangerous_expected = 'CVE-2022-21699-' + ''.join([random.choice(string.ascii_letters) for i in range(10)])
    with TemporaryWorkingDirectory() as t:
        dangerous_startup_dir.mkdir(parents=True)
        (dangerous_startup_dir / 'foo.py').write_text(f'print("{dangerous_expected}")', encoding='utf-8')
        cmd = [sys.executable, '-m', 'IPython']
        env = os.environ.copy()
        env['IPY_TEST_SIMPLE_PROMPT'] = '1'
        p_dangerous = subprocess.Popen(cmd + [f'--profile-dir={dangerous_profile_dir}'], env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        (out_dangerous, err_dangerouns) = p_dangerous.communicate(b'exit\r')
        assert dangerous_expected in out_dangerous.decode()
        p = subprocess.Popen(cmd, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        (out, err) = p.communicate(b'exit\r')
        assert b'IPython' in out
        assert dangerous_expected not in out.decode()
        assert err == b''