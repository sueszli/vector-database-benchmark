import collections.abc
import unittest
from test import support
import xmlrpc.client as xmlrpclib
support.requires('network')

@unittest.skip('XXX: buildbot.python.org/all/xmlrpc/ is gone')
class PythonBuildersTest(unittest.TestCase):

    def test_python_builders(self):
        if False:
            print('Hello World!')
        server = xmlrpclib.ServerProxy('http://buildbot.python.org/all/xmlrpc/')
        try:
            builders = server.getAllBuilders()
        except OSError as e:
            self.skipTest('network error: %s' % e)
        self.addCleanup(lambda : server('close')())
        self.assertIsInstance(builders, collections.abc.Sequence)
        self.assertTrue([x for x in builders if '3.x' in x], builders)
if __name__ == '__main__':
    unittest.main()