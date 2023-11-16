import sys
from plugins.plugin import Plugin

class SSLstripPlus(Plugin):
    name = 'SSLstrip+'
    optname = 'hsts'
    desc = 'Enables SSLstrip+ for partial HSTS bypass'
    version = '0.4'
    tree_info = ['SSLstrip+ by Leonardo Nve running']

    def initialize(self, options):
        if False:
            print('Hello World!')
        self.options = options
        from core.sslstrip.URLMonitor import URLMonitor
        from core.servers.DNS import DNSChef
        from core.utils import iptables
        if iptables().dns is False and options.filter is False:
            iptables().DNS(self.config['MITMf']['DNS']['port'])
        URLMonitor.getInstance().setHstsBypass()
        DNSChef().setHstsBypass()

    def on_shutdown(self):
        if False:
            for i in range(10):
                print('nop')
        from core.utils import iptables
        if iptables().dns is True:
            iptables().flush()