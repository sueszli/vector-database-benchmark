from __future__ import annotations
from shutil import which
from virtualenv.activation import NushellActivator
from virtualenv.info import IS_WIN

def test_nushell(activation_tester_class, activation_tester):
    if False:
        for i in range(10):
            print('nop')

    class Nushell(activation_tester_class):

        def __init__(self, session) -> None:
            if False:
                for i in range(10):
                    print('nop')
            cmd = which('nu')
            if cmd is None and IS_WIN:
                cmd = 'c:\\program files\\nu\\bin\\nu.exe'
            super().__init__(NushellActivator, session, cmd, 'activate.nu', 'nu')
            self.activate_cmd = 'overlay use'
            self.unix_line_ending = not IS_WIN

        def print_prompt(self):
            if False:
                for i in range(10):
                    print('nop')
            return 'print $env.VIRTUAL_PREFIX'

        def activate_call(self, script):
            if False:
                while True:
                    i = 10
            cmd = self.activate_cmd
            scr = self.quote(str(script))
            return f'{cmd} {scr}'.strip()
    activation_tester(Nushell)