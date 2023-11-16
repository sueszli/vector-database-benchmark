"""
Tests for L{twisted.python.fakepwd}.
"""
try:
    import pwd as _pwd
except ImportError:
    pwd = None
else:
    pwd = _pwd
try:
    import spwd as _spwd
except ImportError:
    spwd = None
else:
    spwd = _spwd
import os
from operator import getitem
from twisted.python.compat import _PYPY
from twisted.python.fakepwd import ShadowDatabase, UserDatabase
from twisted.trial.unittest import TestCase
SYSTEM_UID_MAX = 999

def findInvalidUID():
    if False:
        while True:
            i = 10
    "\n    By convention, UIDs less than 1000 are reserved for the system.  A system\n    which allocated every single one of those UIDs would likely have practical\n    problems with allocating new ones, so let's assume that we'll be able to\n    find one.  (If we don't, this will wrap around to negative values and\n    I{eventually} find something.)\n\n    @return: a user ID which does not exist on the local system.  Or, on\n        systems without a L{pwd} module, return C{SYSTEM_UID_MAX}.\n    "
    guess = SYSTEM_UID_MAX
    if pwd is not None:
        while True:
            try:
                pwd.getpwuid(guess)
            except KeyError:
                break
            else:
                guess -= 1
    return guess
INVALID_UID = findInvalidUID()

class UserDatabaseTestsMixin:
    """
    L{UserDatabaseTestsMixin} defines tests which apply to any user database
    implementation.  Subclasses should mix it in, implement C{setUp} to create
    C{self.database} bound to a user database instance, and implement
    C{getExistingUserInfo} to return information about a user (such information
    should be unique per test method).
    """

    def test_getpwuid(self):
        if False:
            print('Hello World!')
        '\n        I{getpwuid} accepts a uid and returns the user record associated with\n        it.\n        '
        for i in range(2):
            (username, password, uid, gid, gecos, dir, shell) = self.getExistingUserInfo()
            entry = self.database.getpwuid(uid)
            self.assertEqual(entry.pw_name, username)
            self.assertEqual(entry.pw_passwd, password)
            self.assertEqual(entry.pw_uid, uid)
            self.assertEqual(entry.pw_gid, gid)
            self.assertEqual(entry.pw_gecos, gecos)
            self.assertEqual(entry.pw_dir, dir)
            self.assertEqual(entry.pw_shell, shell)

    def test_noSuchUID(self):
        if False:
            i = 10
            return i + 15
        '\n        I{getpwuid} raises L{KeyError} when passed a uid which does not exist\n        in the user database.\n        '
        self.assertRaises(KeyError, self.database.getpwuid, INVALID_UID)

    def test_getpwnam(self):
        if False:
            i = 10
            return i + 15
        '\n        I{getpwnam} accepts a username and returns the user record associated\n        with it.\n        '
        for i in range(2):
            (username, password, uid, gid, gecos, dir, shell) = self.getExistingUserInfo()
            entry = self.database.getpwnam(username)
            self.assertEqual(entry.pw_name, username)
            self.assertEqual(entry.pw_passwd, password)
            self.assertEqual(entry.pw_uid, uid)
            self.assertEqual(entry.pw_gid, gid)
            self.assertEqual(entry.pw_gecos, gecos)
            self.assertEqual(entry.pw_dir, dir)
            self.assertEqual(entry.pw_shell, shell)

    def test_getpwnamRejectsBytes(self):
        if False:
            i = 10
            return i + 15
        '\n        L{getpwnam} rejects a non-L{str} username with an exception.\n        '
        exc_type = TypeError
        if _PYPY:
            exc_type = Exception
        self.assertRaises(exc_type, self.database.getpwnam, b'i-am-bytes')

    def test_noSuchName(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        I{getpwnam} raises L{KeyError} when passed a username which does not\n        exist in the user database.\n        '
        self.assertRaises(KeyError, self.database.getpwnam, 'nosuchuserexiststhenameistoolongandhas\x01inittoo')

    def test_recordLength(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        The user record returned by I{getpwuid}, I{getpwnam}, and I{getpwall}\n        has a length.\n        '
        db = self.database
        (username, password, uid, gid, gecos, dir, shell) = self.getExistingUserInfo()
        for entry in [db.getpwuid(uid), db.getpwnam(username), db.getpwall()[0]]:
            self.assertIsInstance(len(entry), int)
            self.assertEqual(len(entry), 7)

    def test_recordIndexable(self):
        if False:
            return 10
        '\n        The user record returned by I{getpwuid}, I{getpwnam}, and I{getpwall}\n        is indexable, with successive indexes starting from 0 corresponding to\n        the values of the C{pw_name}, C{pw_passwd}, C{pw_uid}, C{pw_gid},\n        C{pw_gecos}, C{pw_dir}, and C{pw_shell} attributes, respectively.\n        '
        db = self.database
        (username, password, uid, gid, gecos, dir, shell) = self.getExistingUserInfo()
        for entry in [db.getpwuid(uid), db.getpwnam(username), db.getpwall()[0]]:
            self.assertEqual(entry[0], username)
            self.assertEqual(entry[1], password)
            self.assertEqual(entry[2], uid)
            self.assertEqual(entry[3], gid)
            self.assertEqual(entry[4], gecos)
            self.assertEqual(entry[5], dir)
            self.assertEqual(entry[6], shell)
            self.assertEqual(len(entry), len(list(entry)))
            self.assertRaises(IndexError, getitem, entry, 7)

class UserDatabaseTests(TestCase, UserDatabaseTestsMixin):
    """
    Tests for L{UserDatabase}.
    """

    def setUp(self):
        if False:
            i = 10
            return i + 15
        '\n        Create a L{UserDatabase} with no user data in it.\n        '
        self.database = UserDatabase()
        self._counter = SYSTEM_UID_MAX + 1

    def getExistingUserInfo(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Add a new user to C{self.database} and return its information.\n        '
        self._counter += 1
        suffix = '_' + str(self._counter)
        username = 'username' + suffix
        password = 'password' + suffix
        uid = self._counter
        gid = self._counter + 1000
        gecos = 'gecos' + suffix
        dir = 'dir' + suffix
        shell = 'shell' + suffix
        self.database.addUser(username, password, uid, gid, gecos, dir, shell)
        return (username, password, uid, gid, gecos, dir, shell)

    def test_addUser(self):
        if False:
            print('Hello World!')
        '\n        L{UserDatabase.addUser} accepts seven arguments, one for each field of\n        a L{pwd.struct_passwd}, and makes the new record available via\n        L{UserDatabase.getpwuid}, L{UserDatabase.getpwnam}, and\n        L{UserDatabase.getpwall}.\n        '
        username = 'alice'
        password = 'secr3t'
        uid = 123
        gid = 456
        gecos = 'Alice,,,'
        home = '/users/alice'
        shell = '/usr/bin/foosh'
        db = self.database
        db.addUser(username, password, uid, gid, gecos, home, shell)
        for [entry] in [[db.getpwuid(uid)], [db.getpwnam(username)], db.getpwall()]:
            self.assertEqual(entry.pw_name, username)
            self.assertEqual(entry.pw_passwd, password)
            self.assertEqual(entry.pw_uid, uid)
            self.assertEqual(entry.pw_gid, gid)
            self.assertEqual(entry.pw_gecos, gecos)
            self.assertEqual(entry.pw_dir, home)
            self.assertEqual(entry.pw_shell, shell)

class PwdModuleTests(TestCase, UserDatabaseTestsMixin):
    """
    L{PwdModuleTests} runs the tests defined by L{UserDatabaseTestsMixin}
    against the built-in C{pwd} module.  This serves to verify that
    L{UserDatabase} is really a fake of that API.
    """
    if pwd is None:
        skip = 'Cannot verify UserDatabase against pwd without pwd'
    else:
        database = pwd

    def setUp(self):
        if False:
            while True:
                i = 10
        self._users = iter(self.database.getpwall())
        self._uids = set()

    def getExistingUserInfo(self):
        if False:
            print('Hello World!')
        '\n        Read and return the next record from C{self._users}, filtering out\n        any records with previously seen uid values (as these cannot be\n        found with C{getpwuid} and only cause trouble).\n        '
        while True:
            entry = next(self._users)
            uid = entry.pw_uid
            if uid not in self._uids:
                self._uids.add(uid)
                return entry

class ShadowDatabaseTestsMixin:
    """
    L{ShadowDatabaseTestsMixin} defines tests which apply to any shadow user
    database implementation.  Subclasses should mix it in, implement C{setUp} to
    create C{self.database} bound to a shadow user database instance, and
    implement C{getExistingUserInfo} to return information about a user (such
    information should be unique per test method).
    """

    def test_getspnam(self):
        if False:
            return 10
        '\n        L{getspnam} accepts a username and returns the user record associated\n        with it.\n        '
        for i in range(2):
            (username, password, lastChange, min, max, warn, inact, expire, flag) = self.getExistingUserInfo()
            entry = self.database.getspnam(username)
            self.assertEqual(entry.sp_nam, username)
            self.assertEqual(entry.sp_pwd, password)
            self.assertEqual(entry.sp_lstchg, lastChange)
            self.assertEqual(entry.sp_min, min)
            self.assertEqual(entry.sp_max, max)
            self.assertEqual(entry.sp_warn, warn)
            self.assertEqual(entry.sp_inact, inact)
            self.assertEqual(entry.sp_expire, expire)
            self.assertEqual(entry.sp_flag, flag)

    def test_noSuchName(self):
        if False:
            print('Hello World!')
        '\n        I{getspnam} raises L{KeyError} when passed a username which does not\n        exist in the user database.\n        '
        self.assertRaises(KeyError, self.database.getspnam, 'alice')

    def test_getspnamBytes(self):
        if False:
            print('Hello World!')
        '\n        I{getspnam} raises L{TypeError} when passed a L{bytes}, just like\n        L{spwd.getspnam}.\n        '
        self.assertRaises(TypeError, self.database.getspnam, b'i-am-bytes')

    def test_recordLength(self):
        if False:
            print('Hello World!')
        '\n        The shadow user record returned by I{getspnam} and I{getspall} has a\n        length.\n        '
        db = self.database
        username = self.getExistingUserInfo()[0]
        for entry in [db.getspnam(username), db.getspall()[0]]:
            self.assertIsInstance(len(entry), int)
            self.assertEqual(len(entry), 9)

    def test_recordIndexable(self):
        if False:
            print('Hello World!')
        '\n        The shadow user record returned by I{getpwnam} and I{getspall} is\n        indexable, with successive indexes starting from 0 corresponding to the\n        values of the C{sp_nam}, C{sp_pwd}, C{sp_lstchg}, C{sp_min}, C{sp_max},\n        C{sp_warn}, C{sp_inact}, C{sp_expire}, and C{sp_flag} attributes,\n        respectively.\n        '
        db = self.database
        (username, password, lastChange, min, max, warn, inact, expire, flag) = self.getExistingUserInfo()
        for entry in [db.getspnam(username), db.getspall()[0]]:
            self.assertEqual(entry[0], username)
            self.assertEqual(entry[1], password)
            self.assertEqual(entry[2], lastChange)
            self.assertEqual(entry[3], min)
            self.assertEqual(entry[4], max)
            self.assertEqual(entry[5], warn)
            self.assertEqual(entry[6], inact)
            self.assertEqual(entry[7], expire)
            self.assertEqual(entry[8], flag)
            self.assertEqual(len(entry), len(list(entry)))
            self.assertRaises(IndexError, getitem, entry, 9)

class ShadowDatabaseTests(TestCase, ShadowDatabaseTestsMixin):
    """
    Tests for L{ShadowDatabase}.
    """

    def setUp(self):
        if False:
            print('Hello World!')
        '\n        Create a L{ShadowDatabase} with no user data in it.\n        '
        self.database = ShadowDatabase()
        self._counter = 0

    def getExistingUserInfo(self):
        if False:
            i = 10
            return i + 15
        '\n        Add a new user to C{self.database} and return its information.\n        '
        self._counter += 1
        suffix = '_' + str(self._counter)
        username = 'username' + suffix
        password = 'password' + suffix
        lastChange = self._counter + 1
        min = self._counter + 2
        max = self._counter + 3
        warn = self._counter + 4
        inact = self._counter + 5
        expire = self._counter + 6
        flag = self._counter + 7
        self.database.addUser(username, password, lastChange, min, max, warn, inact, expire, flag)
        return (username, password, lastChange, min, max, warn, inact, expire, flag)

    def test_addUser(self):
        if False:
            i = 10
            return i + 15
        '\n        L{UserDatabase.addUser} accepts seven arguments, one for each field of\n        a L{pwd.struct_passwd}, and makes the new record available via\n        L{UserDatabase.getpwuid}, L{UserDatabase.getpwnam}, and\n        L{UserDatabase.getpwall}.\n        '
        username = 'alice'
        password = 'secr3t'
        lastChange = 17
        min = 42
        max = 105
        warn = 12
        inact = 3
        expire = 400
        flag = 3
        db = self.database
        db.addUser(username, password, lastChange, min, max, warn, inact, expire, flag)
        for [entry] in [[db.getspnam(username)], db.getspall()]:
            self.assertEqual(entry.sp_nam, username)
            self.assertEqual(entry.sp_pwd, password)
            self.assertEqual(entry.sp_lstchg, lastChange)
            self.assertEqual(entry.sp_min, min)
            self.assertEqual(entry.sp_max, max)
            self.assertEqual(entry.sp_warn, warn)
            self.assertEqual(entry.sp_inact, inact)
            self.assertEqual(entry.sp_expire, expire)
            self.assertEqual(entry.sp_flag, flag)

class SPwdModuleTests(TestCase, ShadowDatabaseTestsMixin):
    """
    L{SPwdModuleTests} runs the tests defined by L{ShadowDatabaseTestsMixin}
    against the built-in C{spwd} module.  This serves to verify that
    L{ShadowDatabase} is really a fake of that API.
    """
    if spwd is None:
        skip = 'Cannot verify ShadowDatabase against spwd without spwd'
    elif os.getuid() != 0:
        skip = 'Cannot access shadow user database except as root'
    else:
        database = spwd

    def setUp(self):
        if False:
            while True:
                i = 10
        self._users = iter(self.database.getspall())

    def getExistingUserInfo(self):
        if False:
            print('Hello World!')
        '\n        Read and return the next record from C{self._users}.\n        '
        return next(self._users)