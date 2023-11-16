"""
Tests for implementations of L{inetdconf}.
"""
from twisted.runner import inetdconf
from twisted.trial import unittest

class ServicesConfTests(unittest.TestCase):
    """
    Tests for L{inetdconf.ServicesConf}
    """

    def setUp(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.servicesFilename1 = self.mktemp()
        with open(self.servicesFilename1, 'w') as f:
            f.write('\n            # This is a comment\n            http            80/tcp          www www-http    # WorldWideWeb HTTP\n            http            80/udp          www www-http\n            http            80/sctp\n            ')
        self.servicesFilename2 = self.mktemp()
        with open(self.servicesFilename2, 'w') as f:
            f.write('\n            https           443/tcp                # http protocol over TLS/SSL\n            ')

    def test_parseDefaultFilename(self) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Services are parsed from default filename.\n        '
        conf = inetdconf.ServicesConf()
        conf.defaultFilename = self.servicesFilename1
        conf.parseFile()
        self.assertEqual(conf.services, {('http', 'tcp'): 80, ('http', 'udp'): 80, ('http', 'sctp'): 80, ('www', 'tcp'): 80, ('www', 'udp'): 80, ('www-http', 'tcp'): 80, ('www-http', 'udp'): 80})

    def test_parseFile(self) -> None:
        if False:
            while True:
                i = 10
        '\n        Services are parsed from given C{file}.\n        '
        conf = inetdconf.ServicesConf()
        with open(self.servicesFilename2) as f:
            conf.parseFile(f)
        self.assertEqual(conf.services, {('https', 'tcp'): 443})