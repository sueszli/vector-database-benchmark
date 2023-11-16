import unittest
import threading
import logging

class BasicTests(unittest.TestCase):

    def test_configfile(self):
        if False:
            print('Hello World!')
        from configobj import ConfigObj
        config = ConfigObj('config/mitmf.conf')

    def test_logger(self):
        if False:
            while True:
                i = 10
        from core.logger import logger
        logger.log_level = logging.DEBUG
        formatter = logging.Formatter('%(asctime)s [unittest] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        log = logger().setup_logger('unittest', formatter)

    def test_DNSChef(self):
        if False:
            for i in range(10):
                print('nop')
        from core.logger import logger
        logger.log_level = logging.DEBUG
        from core.servers.DNS import DNSChef
        DNSChef().start()

    def test_NetCreds(self):
        if False:
            print('Hello World!')
        from core.logger import logger
        logger.log_level = logging.DEBUG
        from core.netcreds import NetCreds
        NetCreds().start('venet0:0', '172.30.96.18')

    def test_SSLStrip_Proxy(self):
        if False:
            while True:
                i = 10
        favicon = True
        preserve_cache = True
        killsessions = True
        listen_port = 10000
        from twisted.web import http
        from twisted.internet import reactor
        from core.sslstrip.CookieCleaner import CookieCleaner
        from core.proxyplugins import ProxyPlugins
        from core.sslstrip.StrippingProxy import StrippingProxy
        from core.sslstrip.URLMonitor import URLMonitor
        URLMonitor.getInstance().setFaviconSpoofing(favicon)
        URLMonitor.getInstance().setCaching(preserve_cache)
        CookieCleaner.getInstance().setEnabled(killsessions)
        strippingFactory = http.HTTPFactory(timeout=10)
        strippingFactory.protocol = StrippingProxy
        reactor.listenTCP(listen_port, strippingFactory)
        t = threading.Thread(name='sslstrip_test', target=reactor.run)
        t.setDaemon(True)
        t.start()
if __name__ == '__main__':
    unittest.main()