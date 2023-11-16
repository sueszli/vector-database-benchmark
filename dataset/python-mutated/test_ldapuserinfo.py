import types
from unittest import mock
from twisted.internet import defer
from twisted.trial import unittest
from buildbot.test.reactor import TestReactorMixin
from buildbot.test.util.www import WwwTestMixin
from buildbot.www import avatar
from buildbot.www import ldapuserinfo
try:
    import ldap3
except ImportError:
    ldap3 = None

def get_config_parameter(p):
    if False:
        for i in range(10):
            print('nop')
    params = {'DEFAULT_SERVER_ENCODING': 'utf-8'}
    return params[p]
fake_ldap = types.ModuleType('ldap3')
fake_ldap.SEARCH_SCOPE_WHOLE_SUBTREE = 2
fake_ldap.get_config_parameter = get_config_parameter

class FakeLdap:

    def __init__(self):
        if False:
            while True:
                i = 10

        def search(base, filterstr='f', scope=None, attributes=None):
            if False:
                while True:
                    i = 10
            pass
        self.search = mock.Mock(spec=search)

class CommonTestCase(unittest.TestCase):
    """Common fixture for all ldapuserinfo tests

    we completely fake the ldap3 module, so no need to require
    it to run the unit tests
    """
    if not ldap3:
        skip = 'ldap3 is required for LdapUserInfo tests'

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.ldap = FakeLdap()
        self.makeUserInfoProvider()
        self.userInfoProvider.connectLdap = lambda : self.ldap

        def search(base, filterstr='f', attributes=None):
            if False:
                for i in range(10):
                    print('nop')
            pass
        self.userInfoProvider.search = mock.Mock(spec=search)

    def makeUserInfoProvider(self):
        if False:
            return 10
        'To be implemented by subclasses'
        raise NotImplementedError

    def _makeSearchSideEffect(self, attribute_type, ret):
        if False:
            print('Hello World!')
        ret = [[{'dn': i[0], attribute_type: i[1]} for i in r] for r in ret]
        self.userInfoProvider.search.side_effect = ret

    def makeSearchSideEffect(self, ret):
        if False:
            for i in range(10):
                print('nop')
        return self._makeSearchSideEffect('attributes', ret)

    def makeRawSearchSideEffect(self, ret):
        if False:
            i = 10
            return i + 15
        return self._makeSearchSideEffect('raw_attributes', ret)

    def assertSearchCalledWith(self, exp):
        if False:
            i = 10
            return i + 15
        got = self.userInfoProvider.search.call_args_list
        self.assertEqual(len(exp), len(got))
        for (i, val) in enumerate(exp):
            self.assertEqual(val[0][0], got[i][0][1])
            self.assertEqual(val[0][1], got[i][0][2])
            self.assertEqual(val[0][2], got[i][1]['attributes'])

class LdapUserInfo(CommonTestCase):

    def makeUserInfoProvider(self):
        if False:
            return 10
        self.userInfoProvider = ldapuserinfo.LdapUserInfo(uri='ldap://uri', bindUser='user', bindPw='pass', accountBase='accbase', groupBase='groupbase', accountPattern='accpattern', groupMemberPattern='groupMemberPattern', accountFullName='accountFullName', accountEmail='accountEmail', groupName='groupName', avatarPattern='avatar', avatarData='picture', accountExtraFields=['myfield'])

    @defer.inlineCallbacks
    def test_updateUserInfoNoResults(self):
        if False:
            while True:
                i = 10
        self.makeSearchSideEffect([[], [], []])
        try:
            yield self.userInfoProvider.getUserInfo('me')
        except KeyError as e:
            self.assertRegex(repr(e), 'KeyError\\(\'ldap search \\"accpattern\\" returned 0 results\',?\\)')
        else:
            self.fail('should have raised a key error')

    @defer.inlineCallbacks
    def test_updateUserInfoNoGroups(self):
        if False:
            while True:
                i = 10
        self.makeSearchSideEffect([[('cn', {'accountFullName': 'me too', 'accountEmail': 'mee@too'})], [], []])
        res = (yield self.userInfoProvider.getUserInfo('me'))
        self.assertSearchCalledWith([(('accbase', 'accpattern', ['accountEmail', 'accountFullName', 'myfield']), {}), (('groupbase', 'groupMemberPattern', ['groupName']), {})])
        self.assertEqual(res, {'email': 'mee@too', 'full_name': 'me too', 'groups': [], 'username': 'me'})

    @defer.inlineCallbacks
    def test_updateUserInfoGroups(self):
        if False:
            print('Hello World!')
        self.makeSearchSideEffect([[('cn', {'accountFullName': 'me too', 'accountEmail': 'mee@too'})], [('cn', {'groupName': ['group']}), ('cn', {'groupName': ['group2']})], []])
        res = (yield self.userInfoProvider.getUserInfo('me'))
        self.assertEqual(res, {'email': 'mee@too', 'full_name': 'me too', 'groups': ['group', 'group2'], 'username': 'me'})

    @defer.inlineCallbacks
    def test_updateUserInfoGroupsUnicodeDn(self):
        if False:
            for i in range(10):
                print('nop')
        dn = 'cn=Sébastien,dc=example,dc=org'
        self.userInfoProvider.groupMemberPattern = '(member=%(dn)s)'
        self.makeSearchSideEffect([[(dn, {'accountFullName': 'me too', 'accountEmail': 'mee@too'})], [('cn', {'groupName': ['group']}), ('cn', {'groupName': ['group2']})], []])
        res = (yield self.userInfoProvider.getUserInfo('me'))
        self.assertEqual(res, {'email': 'mee@too', 'full_name': 'me too', 'groups': ['group', 'group2'], 'username': 'me'})

class LdapAvatar(CommonTestCase, TestReactorMixin, WwwTestMixin):

    @defer.inlineCallbacks
    def setUp(self):
        if False:
            i = 10
            return i + 15
        CommonTestCase.setUp(self)
        self.setup_test_reactor()
        master = self.make_master(url='http://a/b/', avatar_methods=[self.userInfoProvider])
        self.rsrc = avatar.AvatarResource(master)
        self.rsrc.reconfigResource(master.config)
        yield self.master.startService()

    def makeUserInfoProvider(self):
        if False:
            return 10
        self.userInfoProvider = ldapuserinfo.LdapUserInfo(uri='ldap://uri', bindUser='user', bindPw='pass', accountBase='accbase', groupBase='groupbase', accountPattern='accpattern=%(username)s', groupMemberPattern='groupMemberPattern', accountFullName='accountFullName', accountEmail='accountEmail', groupName='groupName', avatarPattern='avatar=%(email)s', avatarData='picture', accountExtraFields=['myfield'])

    @defer.inlineCallbacks
    def _getUserAvatar(self, mimeTypeAndData):
        if False:
            for i in range(10):
                print('nop')
        (_, data) = mimeTypeAndData
        self.makeRawSearchSideEffect([[('cn', {'picture': [data]})]])
        res = (yield self.render_resource(self.rsrc, b'/?email=me'))
        self.assertSearchCalledWith([(('accbase', 'avatar=me', ['picture']), {})])
        return res

    @defer.inlineCallbacks
    def test_getUserAvatarPNG(self):
        if False:
            return 10
        mimeTypeAndData = (b'image/png', b'\x89PNG lljklj')
        yield self._getUserAvatar(mimeTypeAndData)
        self.assertRequest(contentType=mimeTypeAndData[0], content=mimeTypeAndData[1])

    @defer.inlineCallbacks
    def test_getUserAvatarJPEG(self):
        if False:
            print('Hello World!')
        mimeTypeAndData = (b'image/jpeg', b'\xff\xd8\xff lljklj')
        yield self._getUserAvatar(mimeTypeAndData)
        self.assertRequest(contentType=mimeTypeAndData[0], content=mimeTypeAndData[1])

    @defer.inlineCallbacks
    def test_getUserAvatarGIF(self):
        if False:
            while True:
                i = 10
        mimeTypeAndData = (b'image/gif', b'GIF8 lljklj')
        yield self._getUserAvatar(mimeTypeAndData)
        self.assertRequest(contentType=mimeTypeAndData[0], content=mimeTypeAndData[1])

    @defer.inlineCallbacks
    def test_getUserAvatarUnknownType(self):
        if False:
            while True:
                i = 10
        mimeTypeAndData = (b'', b'unknown image format')
        res = (yield self._getUserAvatar(mimeTypeAndData))
        self.assertEqual(res, {'redirected': b'img/nobody.png'})

    @defer.inlineCallbacks
    def test_getUsernameAvatar(self):
        if False:
            i = 10
            return i + 15
        mimeType = b'image/gif'
        data = b'GIF8 lljklj'
        self.makeRawSearchSideEffect([[('cn', {'picture': [data]})]])
        yield self.render_resource(self.rsrc, b'/?username=me')
        self.assertSearchCalledWith([(('accbase', 'accpattern=me', ['picture']), {})])
        self.assertRequest(contentType=mimeType, content=data)

    @defer.inlineCallbacks
    def test_getUnknownUsernameAvatar(self):
        if False:
            for i in range(10):
                print('nop')
        self.makeSearchSideEffect([[], [], []])
        res = (yield self.render_resource(self.rsrc, b'/?username=other'))
        self.assertSearchCalledWith([(('accbase', 'accpattern=other', ['picture']), {})])
        self.assertEqual(res, {'redirected': b'img/nobody.png'})

class LdapUserInfoNotEscCharsDn(CommonTestCase):

    def makeUserInfoProvider(self):
        if False:
            for i in range(10):
                print('nop')
        self.userInfoProvider = ldapuserinfo.LdapUserInfo(uri='ldap://uri', bindUser='user', bindPw='pass', accountBase='accbase', groupBase='groupbase', accountPattern='accpattern', groupMemberPattern='(member=%(dn)s)', accountFullName='accountFullName', accountEmail='accountEmail', groupName='groupName', avatarPattern='avatar', avatarData='picture')

    @defer.inlineCallbacks
    def test_getUserInfoGroupsNotEscCharsDn(self):
        if False:
            return 10
        dn = 'cn=Lastname, Firstname \x028UIDxxx\x029,dc=example,dc=org'
        pattern = self.userInfoProvider.groupMemberPattern % {'dn': dn}
        self.makeSearchSideEffect([[(dn, {'accountFullName': 'Lastname, Firstname (UIDxxx)', 'accountEmail': 'mee@too'})], [('cn', {'groupName': ['group']}), ('cn', {'groupName': ['group2']})], []])
        res = (yield self.userInfoProvider.getUserInfo('me'))
        self.assertSearchCalledWith([(('accbase', 'accpattern', ['accountEmail', 'accountFullName']), {}), (('groupbase', pattern, ['groupName']), {})])
        self.assertEqual(res, {'email': 'mee@too', 'full_name': 'Lastname, Firstname (UIDxxx)', 'groups': ['group', 'group2'], 'username': 'me'})

class LdapUserInfoNoGroups(CommonTestCase):

    def makeUserInfoProvider(self):
        if False:
            print('Hello World!')
        self.userInfoProvider = ldapuserinfo.LdapUserInfo(uri='ldap://uri', bindUser='user', bindPw='pass', accountBase='accbase', accountPattern='accpattern', accountFullName='accountFullName', accountEmail='accountEmail', avatarPattern='avatar', avatarData='picture', accountExtraFields=['myfield'])

    @defer.inlineCallbacks
    def test_updateUserInfo(self):
        if False:
            while True:
                i = 10
        self.makeSearchSideEffect([[('cn', {'accountFullName': 'me too', 'accountEmail': 'mee@too'})], [], []])
        res = (yield self.userInfoProvider.getUserInfo('me'))
        self.assertSearchCalledWith([(('accbase', 'accpattern', ['accountEmail', 'accountFullName', 'myfield']), {})])
        self.assertEqual(res, {'email': 'mee@too', 'full_name': 'me too', 'groups': [], 'username': 'me'})

class Config(unittest.TestCase):
    if not ldap3:
        skip = 'ldap3 is required for LdapUserInfo tests'

    def test_missing_group_name(self):
        if False:
            for i in range(10):
                print('nop')
        with self.assertRaises(ValueError):
            ldapuserinfo.LdapUserInfo(groupMemberPattern='member=%(dn)s', groupBase='grpbase', uri='ldap://uri', bindUser='user', bindPw='pass', accountBase='accbase', accountPattern='accpattern', accountFullName='accountFullName', accountEmail='accountEmail')

    def test_missing_group_base(self):
        if False:
            for i in range(10):
                print('nop')
        with self.assertRaises(ValueError):
            ldapuserinfo.LdapUserInfo(groupMemberPattern='member=%(dn)s', groupName='group', uri='ldap://uri', bindUser='user', bindPw='pass', accountBase='accbase', accountPattern='accpattern', accountFullName='accountFullName', accountEmail='accountEmail')

    def test_missing_two_params(self):
        if False:
            return 10
        with self.assertRaises(ValueError):
            ldapuserinfo.LdapUserInfo(groupName='group', uri='ldap://uri', bindUser='user', bindPw='pass', accountBase='accbase', accountPattern='accpattern', accountFullName='accountFullName', accountEmail='accountEmail')