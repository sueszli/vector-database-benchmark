from astropy.samp import conf
from astropy.samp.hub import SAMPHubServer
from astropy.samp.hub_proxy import SAMPHubProxy

def setup_module(module):
    if False:
        return 10
    conf.use_internet = False

class TestHubProxy:

    def setup_method(self, method):
        if False:
            i = 10
            return i + 15
        self.hub = SAMPHubServer(web_profile=False, mode='multiple', pool_size=1)
        self.hub.start()
        self.proxy = SAMPHubProxy()
        self.proxy.connect(hub=self.hub, pool_size=1)

    def teardown_method(self, method):
        if False:
            print('Hello World!')
        if self.proxy.is_connected:
            self.proxy.disconnect()
        self.hub.stop()

    def test_is_connected(self):
        if False:
            for i in range(10):
                print('nop')
        assert self.proxy.is_connected

    def test_disconnect(self):
        if False:
            print('Hello World!')
        self.proxy.disconnect()

    def test_ping(self):
        if False:
            for i in range(10):
                print('nop')
        self.proxy.ping()

    def test_registration(self):
        if False:
            return 10
        result = self.proxy.register(self.proxy.lockfile['samp.secret'])
        self.proxy.unregister(result['samp.private-key'])

def test_custom_lockfile(tmp_path):
    if False:
        return 10
    lockfile = str(tmp_path / '.samptest')
    hub = SAMPHubServer(web_profile=False, lockfile=lockfile, pool_size=1)
    hub.start()
    proxy = SAMPHubProxy()
    proxy.connect(hub=hub, pool_size=1)
    hub.stop()