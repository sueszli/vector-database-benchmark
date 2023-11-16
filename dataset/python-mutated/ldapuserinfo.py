import importlib
from urllib.parse import urlparse
from twisted.internet import threads
from buildbot.util import bytes2unicode
from buildbot.util import flatten
from buildbot.www import auth
from buildbot.www import avatar
try:
    import ldap3
except ImportError:
    ldap3 = None

class LdapUserInfo(avatar.AvatarBase, auth.UserInfoProviderBase):
    name = 'ldap'

    def __init__(self, uri, bindUser, bindPw, accountBase, accountPattern, accountFullName, accountEmail, groupBase=None, groupMemberPattern=None, groupName=None, avatarPattern=None, avatarData=None, accountExtraFields=None, tls=None):
        if False:
            while True:
                i = 10
        if not ldap3:
            importlib.import_module('ldap3')
        self.uri = uri
        self.bindUser = bindUser
        self.bindPw = bindPw
        self.accountBase = accountBase
        self.accountEmail = accountEmail
        self.accountPattern = accountPattern
        self.accountFullName = accountFullName
        group_params = [p for p in (groupName, groupMemberPattern, groupBase) if p is not None]
        if len(group_params) not in (0, 3):
            raise ValueError('Incomplete LDAP groups configuration. To use Ldap groups, you need to specify the three parameters (groupName, groupMemberPattern and groupBase). ')
        self.groupName = groupName
        self.groupMemberPattern = groupMemberPattern
        self.groupBase = groupBase
        self.avatarPattern = avatarPattern
        self.avatarData = avatarData
        if accountExtraFields is None:
            accountExtraFields = []
        self.accountExtraFields = accountExtraFields
        self.ldap_encoding = ldap3.get_config_parameter('DEFAULT_SERVER_ENCODING')
        self.tls = tls

    def connectLdap(self):
        if False:
            print('Hello World!')
        server = urlparse(self.uri)
        netloc = server.netloc.split(':')
        s = ldap3.Server(netloc[0], port=int(netloc[1]), use_ssl=server.scheme == 'ldaps', get_info=ldap3.ALL, tls=self.tls)
        auth = ldap3.SIMPLE
        if self.bindUser is None and self.bindPw is None:
            auth = ldap3.ANONYMOUS
        c = ldap3.Connection(s, auto_bind=True, client_strategy=ldap3.SYNC, user=self.bindUser, password=self.bindPw, authentication=auth)
        return c

    def search(self, c, base, filterstr='f', attributes=None):
        if False:
            for i in range(10):
                print('nop')
        c.search(base, filterstr, ldap3.SUBTREE, attributes=attributes)
        return c.response

    def getUserInfo(self, username):
        if False:
            for i in range(10):
                print('nop')
        username = bytes2unicode(username)

        def thd():
            if False:
                print('Hello World!')
            c = self.connectLdap()
            infos = {'username': username}
            pattern = self.accountPattern % {'username': username}
            res = self.search(c, self.accountBase, pattern, attributes=[self.accountEmail, self.accountFullName] + self.accountExtraFields)
            if len(res) != 1:
                raise KeyError(f'ldap search "{pattern}" returned {len(res)} results')
            (dn, ldap_infos) = (res[0]['dn'], res[0]['attributes'])

            def getFirstLdapInfo(x):
                if False:
                    return 10
                if isinstance(x, list):
                    x = x[0] if x else None
                return x
            infos['full_name'] = getFirstLdapInfo(ldap_infos[self.accountFullName])
            infos['email'] = getFirstLdapInfo(ldap_infos[self.accountEmail])
            for f in self.accountExtraFields:
                if f in ldap_infos:
                    infos[f] = getFirstLdapInfo(ldap_infos[f])
            if self.groupMemberPattern is None:
                infos['groups'] = []
                return infos
            pattern = self.groupMemberPattern % {'dn': ldap3.utils.conv.escape_filter_chars(dn)}
            res = self.search(c, self.groupBase, pattern, attributes=[self.groupName])
            infos['groups'] = flatten([group_infos['attributes'][self.groupName] for group_infos in res])
            return infos
        return threads.deferToThread(thd)

    def findAvatarMime(self, data):
        if False:
            for i in range(10):
                print('nop')
        if data.startswith(b'\xff\xd8\xff'):
            return (b'image/jpeg', data)
        if data.startswith(b'\x89PNG'):
            return (b'image/png', data)
        if data.startswith(b'GIF8'):
            return (b'image/gif', data)
        return None

    def getUserAvatar(self, email, username, size, defaultAvatarUrl):
        if False:
            i = 10
            return i + 15
        if username:
            username = bytes2unicode(username)
        if email:
            email = bytes2unicode(email)

        def thd():
            if False:
                i = 10
                return i + 15
            c = self.connectLdap()
            if username:
                pattern = self.accountPattern % {'username': username}
            elif email:
                pattern = self.avatarPattern % {'email': email}
            else:
                return None
            res = self.search(c, self.accountBase, pattern, attributes=[self.avatarData])
            if not res:
                return None
            ldap_infos = res[0]['raw_attributes']
            if self.avatarData in ldap_infos and ldap_infos[self.avatarData]:
                data = ldap_infos[self.avatarData][0]
                return self.findAvatarMime(data)
            return None
        return threads.deferToThread(thd)