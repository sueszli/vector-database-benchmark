from . import testutils
import unittest
import psycopg2
import psycopg2.extras
try:
    import ipaddress as ip
except ImportError:
    ip = None

@unittest.skipIf(ip is None, "'ipaddress' module not available")
class NetworkingTestCase(testutils.ConnectingTestCase):

    def test_inet_cast(self):
        if False:
            print('Hello World!')
        cur = self.conn.cursor()
        psycopg2.extras.register_ipaddress(cur)
        cur.execute('select null::inet')
        self.assert_(cur.fetchone()[0] is None)
        cur.execute("select '127.0.0.1/24'::inet")
        obj = cur.fetchone()[0]
        self.assert_(isinstance(obj, ip.IPv4Interface), repr(obj))
        self.assertEquals(obj, ip.ip_interface('127.0.0.1/24'))
        cur.execute("select '::ffff:102:300/128'::inet")
        obj = cur.fetchone()[0]
        self.assert_(isinstance(obj, ip.IPv6Interface), repr(obj))
        self.assertEquals(obj, ip.ip_interface('::ffff:102:300/128'))

    @testutils.skip_before_postgres(8, 2)
    def test_inet_array_cast(self):
        if False:
            for i in range(10):
                print('nop')
        cur = self.conn.cursor()
        psycopg2.extras.register_ipaddress(cur)
        cur.execute("select '{NULL,127.0.0.1,::ffff:102:300/128}'::inet[]")
        l = cur.fetchone()[0]
        self.assert_(l[0] is None)
        self.assertEquals(l[1], ip.ip_interface('127.0.0.1'))
        self.assertEquals(l[2], ip.ip_interface('::ffff:102:300/128'))
        self.assert_(isinstance(l[1], ip.IPv4Interface), l)
        self.assert_(isinstance(l[2], ip.IPv6Interface), l)

    def test_inet_adapt(self):
        if False:
            i = 10
            return i + 15
        cur = self.conn.cursor()
        psycopg2.extras.register_ipaddress(cur)
        cur.execute('select %s', [ip.ip_interface('127.0.0.1/24')])
        self.assertEquals(cur.fetchone()[0], '127.0.0.1/24')
        cur.execute('select %s', [ip.ip_interface('::ffff:102:300/128')])
        self.assertEquals(cur.fetchone()[0], '::ffff:102:300/128')

    @testutils.skip_if_crdb('cidr')
    def test_cidr_cast(self):
        if False:
            print('Hello World!')
        cur = self.conn.cursor()
        psycopg2.extras.register_ipaddress(cur)
        cur.execute('select null::cidr')
        self.assert_(cur.fetchone()[0] is None)
        cur.execute("select '127.0.0.0/24'::cidr")
        obj = cur.fetchone()[0]
        self.assert_(isinstance(obj, ip.IPv4Network), repr(obj))
        self.assertEquals(obj, ip.ip_network('127.0.0.0/24'))
        cur.execute("select '::ffff:102:300/128'::cidr")
        obj = cur.fetchone()[0]
        self.assert_(isinstance(obj, ip.IPv6Network), repr(obj))
        self.assertEquals(obj, ip.ip_network('::ffff:102:300/128'))

    @testutils.skip_if_crdb('cidr')
    @testutils.skip_before_postgres(8, 2)
    def test_cidr_array_cast(self):
        if False:
            i = 10
            return i + 15
        cur = self.conn.cursor()
        psycopg2.extras.register_ipaddress(cur)
        cur.execute("select '{NULL,127.0.0.1,::ffff:102:300/128}'::cidr[]")
        l = cur.fetchone()[0]
        self.assert_(l[0] is None)
        self.assertEquals(l[1], ip.ip_network('127.0.0.1'))
        self.assertEquals(l[2], ip.ip_network('::ffff:102:300/128'))
        self.assert_(isinstance(l[1], ip.IPv4Network), l)
        self.assert_(isinstance(l[2], ip.IPv6Network), l)

    def test_cidr_adapt(self):
        if False:
            print('Hello World!')
        cur = self.conn.cursor()
        psycopg2.extras.register_ipaddress(cur)
        cur.execute('select %s', [ip.ip_network('127.0.0.0/24')])
        self.assertEquals(cur.fetchone()[0], '127.0.0.0/24')
        cur.execute('select %s', [ip.ip_network('::ffff:102:300/128')])
        self.assertEquals(cur.fetchone()[0], '::ffff:102:300/128')

def test_suite():
    if False:
        print('Hello World!')
    return unittest.TestLoader().loadTestsFromName(__name__)
if __name__ == '__main__':
    unittest.main()