from __future__ import absolute_import
import unittest2
from st2common.util.ip_utils import split_host_port

class IPUtilsTests(unittest2.TestCase):

    def test_host_port_split(self):
        if False:
            while True:
                i = 10
        host_str = '1.2.3.4'
        (host, port) = split_host_port(host_str)
        self.assertEqual(host, host_str)
        self.assertEqual(port, None)
        host_str = '1.2.3.4:55'
        (host, port) = split_host_port(host_str)
        self.assertEqual(host, '1.2.3.4')
        self.assertEqual(port, 55)
        host_str = 'fec2::10'
        (host, port) = split_host_port(host_str)
        self.assertEqual(host, 'fec2::10')
        self.assertEqual(port, None)
        host_str = '[fec2::10]'
        (host, port) = split_host_port(host_str)
        self.assertEqual(host, 'fec2::10')
        self.assertEqual(port, None)
        host_str = '[fec2::10]:55'
        (host, port) = split_host_port(host_str)
        self.assertEqual(host, 'fec2::10')
        self.assertEqual(port, 55)
        host_str = '[1.2.3.4]'
        (host, port) = split_host_port(host_str)
        self.assertEqual(host, '1.2.3.4')
        self.assertEqual(port, None)
        host_str = '[1.2.3.4]:55'
        (host, port) = split_host_port(host_str)
        self.assertEqual(host, '1.2.3.4')
        self.assertEqual(port, 55)
        host_str = '[st2build001]:55'
        (host, port) = split_host_port(host_str)
        self.assertEqual(host, 'st2build001')
        self.assertEqual(port, 55)
        host_str = 'st2build001'
        (host, port) = split_host_port(host_str)
        self.assertEqual(host, 'st2build001')
        self.assertEqual(port, None)
        host_str = 'st2build001:55'
        (host, port) = split_host_port(host_str)
        self.assertEqual(host, 'st2build001')
        self.assertEqual(port, 55)
        host_str = 'st2build001:abc'
        self.assertRaises(Exception, split_host_port, host_str)
        host_str = '[fec2::10]:abc'
        self.assertRaises(Exception, split_host_port, host_str)