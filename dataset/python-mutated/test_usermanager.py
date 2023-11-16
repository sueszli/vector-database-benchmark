"""
Unit tests for octoprint.access.users.UserManager
"""
__license__ = 'GNU Affero General Public License http://www.gnu.org/licenses/agpl.html'
__copyright__ = 'Copyright (C) 2017 The OctoPrint Project - Released under terms of the AGPLv3 License'
import unittest
import ddt
import octoprint.access.users

@ddt.ddt
class UserManagerTest(unittest.TestCase):

    def test_createPasswordHash_nonascii(self):
        if False:
            while True:
                i = 10
        'Test for issue #1891'
        password = 'password with ümläutß'
        octoprint.access.users.UserManager.create_password_hash(password)

    def test_createPasswordHash_is_valid(self):
        if False:
            i = 10
            return i + 15
        password = 'test1234'
        password_hash = octoprint.access.users.UserManager.create_password_hash(password)
        user = octoprint.access.users.User('username', password_hash, True, permissions=[], apikey='apikey', settings={'key': 'value'})
        self.assertTrue(user.check_password(password))