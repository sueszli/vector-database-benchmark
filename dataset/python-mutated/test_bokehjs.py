from __future__ import annotations
import pytest
pytest
import os
import subprocess

class TestBokehJS:

    def test_bokehjs(self) -> None:
        if False:
            return 10
        os.chdir('bokehjs')
        proc = subprocess.Popen(['node', 'make', 'test'], stdout=subprocess.PIPE)
        (out, _) = proc.communicate()
        msg = out.decode('utf-8', errors='ignore')
        os.chdir('..')
        print(msg)
        if proc.returncode != 0:
            assert False