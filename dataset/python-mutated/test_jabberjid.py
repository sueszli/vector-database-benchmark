"""
Tests for L{twisted.words.protocols.jabber.jid}.
"""
from twisted.trial import unittest
from twisted.words.protocols.jabber import jid

class JIDParsingTests(unittest.TestCase):

    def test_parse(self) -> None:
        if False:
            return 10
        '\n        Test different forms of JIDs.\n        '
        self.assertEqual(jid.parse('user@host/resource'), ('user', 'host', 'resource'))
        self.assertEqual(jid.parse('user@host'), ('user', 'host', None))
        self.assertEqual(jid.parse('host'), (None, 'host', None))
        self.assertEqual(jid.parse('host/resource'), (None, 'host', 'resource'))
        self.assertEqual(jid.parse('foo/bar@baz'), (None, 'foo', 'bar@baz'))
        self.assertEqual(jid.parse('boo@foo/bar@baz'), ('boo', 'foo', 'bar@baz'))
        self.assertEqual(jid.parse('boo@foo/bar/baz'), ('boo', 'foo', 'bar/baz'))
        self.assertEqual(jid.parse('boo/foo@bar@baz'), (None, 'boo', 'foo@bar@baz'))
        self.assertEqual(jid.parse('boo/foo/bar'), (None, 'boo', 'foo/bar'))
        self.assertEqual(jid.parse('boo//foo'), (None, 'boo', '/foo'))

    def test_noHost(self) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Test for failure on no host part.\n        '
        self.assertRaises(jid.InvalidFormat, jid.parse, 'user@')

    def test_doubleAt(self) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Test for failure on double @ signs.\n\n        This should fail because @ is not a valid character for the host\n        part of the JID.\n        '
        self.assertRaises(jid.InvalidFormat, jid.parse, 'user@@host')

    def test_multipleAt(self) -> None:
        if False:
            while True:
                i = 10
        '\n        Test for failure on two @ signs.\n\n        This should fail because @ is not a valid character for the host\n        part of the JID.\n        '
        self.assertRaises(jid.InvalidFormat, jid.parse, 'user@host@host')

    def test_prepCaseMapUser(self) -> None:
        if False:
            return 10
        '\n        Test case mapping of the user part of the JID.\n        '
        self.assertEqual(jid.prep('UsEr', 'host', 'resource'), ('user', 'host', 'resource'))

    def test_prepCaseMapHost(self) -> None:
        if False:
            while True:
                i = 10
        '\n        Test case mapping of the host part of the JID.\n        '
        self.assertEqual(jid.prep('user', 'hoST', 'resource'), ('user', 'host', 'resource'))

    def test_prepNoCaseMapResource(self) -> None:
        if False:
            print('Hello World!')
        '\n        Test no case mapping of the resourcce part of the JID.\n        '
        self.assertEqual(jid.prep('user', 'hoST', 'resource'), ('user', 'host', 'resource'))
        self.assertNotEqual(jid.prep('user', 'host', 'Resource'), ('user', 'host', 'resource'))

class JIDTests(unittest.TestCase):

    def test_noneArguments(self) -> None:
        if False:
            print('Hello World!')
        '\n        Test that using no arguments raises an exception.\n        '
        self.assertRaises(RuntimeError, jid.JID)

    def test_attributes(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Test that the attributes correspond with the JID parts.\n        '
        j = jid.JID('user@host/resource')
        self.assertEqual(j.user, 'user')
        self.assertEqual(j.host, 'host')
        self.assertEqual(j.resource, 'resource')

    def test_userhost(self) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Test the extraction of the bare JID.\n        '
        j = jid.JID('user@host/resource')
        self.assertEqual('user@host', j.userhost())

    def test_userhostOnlyHost(self) -> None:
        if False:
            while True:
                i = 10
        '\n        Test the extraction of the bare JID of the full form host/resource.\n        '
        j = jid.JID('host/resource')
        self.assertEqual('host', j.userhost())

    def test_userhostJID(self) -> None:
        if False:
            return 10
        '\n        Test getting a JID object of the bare JID.\n        '
        j1 = jid.JID('user@host/resource')
        j2 = jid.internJID('user@host')
        self.assertIdentical(j2, j1.userhostJID())

    def test_userhostJIDNoResource(self) -> None:
        if False:
            while True:
                i = 10
        '\n        Test getting a JID object of the bare JID when there was no resource.\n        '
        j = jid.JID('user@host')
        self.assertIdentical(j, j.userhostJID())

    def test_fullHost(self) -> None:
        if False:
            while True:
                i = 10
        '\n        Test giving a string representation of the JID with only a host part.\n        '
        j = jid.JID(tuple=(None, 'host', None))
        self.assertEqual('host', j.full())

    def test_fullHostResource(self) -> None:
        if False:
            print('Hello World!')
        '\n        Test giving a string representation of the JID with host, resource.\n        '
        j = jid.JID(tuple=(None, 'host', 'resource'))
        self.assertEqual('host/resource', j.full())

    def test_fullUserHost(self) -> None:
        if False:
            while True:
                i = 10
        '\n        Test giving a string representation of the JID with user, host.\n        '
        j = jid.JID(tuple=('user', 'host', None))
        self.assertEqual('user@host', j.full())

    def test_fullAll(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Test giving a string representation of the JID.\n        '
        j = jid.JID(tuple=('user', 'host', 'resource'))
        self.assertEqual('user@host/resource', j.full())

    def test_equality(self) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Test JID equality.\n        '
        j1 = jid.JID('user@host/resource')
        j2 = jid.JID('user@host/resource')
        self.assertNotIdentical(j1, j2)
        self.assertEqual(j1, j2)

    def test_equalityWithNonJIDs(self) -> None:
        if False:
            while True:
                i = 10
        '\n        Test JID equality.\n        '
        j = jid.JID('user@host/resource')
        self.assertFalse(j == 'user@host/resource')

    def test_inequality(self) -> None:
        if False:
            while True:
                i = 10
        '\n        Test JID inequality.\n        '
        j1 = jid.JID('user1@host/resource')
        j2 = jid.JID('user2@host/resource')
        self.assertNotEqual(j1, j2)

    def test_inequalityWithNonJIDs(self) -> None:
        if False:
            return 10
        '\n        Test JID equality.\n        '
        j = jid.JID('user@host/resource')
        self.assertNotEqual(j, 'user@host/resource')

    def test_hashable(self) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Test JID hashability.\n        '
        j1 = jid.JID('user@host/resource')
        j2 = jid.JID('user@host/resource')
        self.assertEqual(hash(j1), hash(j2))

    def test_str(self) -> None:
        if False:
            print('Hello World!')
        '\n        Test unicode representation of JIDs.\n        '
        j = jid.JID(tuple=('user', 'host', 'resource'))
        self.assertEqual('user@host/resource', str(j))

    def test_repr(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Test representation of JID objects.\n        '
        j = jid.JID(tuple=('user', 'host', 'resource'))
        self.assertEqual('JID(%s)' % repr('user@host/resource'), repr(j))

class InternJIDTests(unittest.TestCase):

    def test_identity(self) -> None:
        if False:
            print('Hello World!')
        '\n        Test that two interned JIDs yield the same object.\n        '
        j1 = jid.internJID('user@host')
        j2 = jid.internJID('user@host')
        self.assertIdentical(j1, j2)