import os
from random import randint
from ulauncher.modes.shortcuts.run_script import run_script

class TestRunScriptAction:

    def test_run_with_arg(self):
        if False:
            i = 10
            return i + 15
        test_file = f'/tmp/ulauncher_test_{randint(1, 111111)}'
        arg = 'hello world'
        thread = run_script(f'#!/bin/bash\necho $@ > {test_file}', arg)
        thread.join()
        with open(test_file) as f:
            assert f.read() == f'{arg}\n'
        os.remove(test_file)