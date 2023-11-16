from __future__ import annotations
from virtualenv.activation import CShellActivator

def test_csh(activation_tester_class, activation_tester):
    if False:
        while True:
            i = 10

    class Csh(activation_tester_class):

        def __init__(self, session) -> None:
            if False:
                i = 10
                return i + 15
            super().__init__(CShellActivator, session, 'csh', 'activate.csh', 'csh')

        def print_prompt(self):
            if False:
                for i in range(10):
                    print('nop')
            return 'echo \'source "$VIRTUAL_ENV/bin/activate.csh"; echo $prompt\' | csh -i ; echo'
    activation_tester(Csh)