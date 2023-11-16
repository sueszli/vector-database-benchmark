"""
This module allows you to manage assistive access on macOS minions with 10.9+

.. versionadded:: 2016.3.0

.. code-block:: bash

    salt '*' assistive.install /usr/bin/osascript
"""
import hashlib
import logging
import sqlite3
import salt.utils.platform
import salt.utils.stringutils
from salt.exceptions import CommandExecutionError
from salt.utils.versions import Version
log = logging.getLogger(__name__)
__virtualname__ = 'assistive'
__func_alias__ = {'enable_': 'enable'}
TCC_DB_PATH = '/Library/Application Support/com.apple.TCC/TCC.db'

def __virtual__():
    if False:
        return 10
    '\n    Only work on Mac OS\n    '
    if not salt.utils.platform.is_darwin():
        return (False, 'Must be run on macOS')
    if Version(__grains__['osrelease']) < Version('10.9'):
        return (False, 'Must be run on macOS 10.9 or newer')
    return __virtualname__

def install(app_id, enable=True):
    if False:
        return 10
    "\n    Install a bundle ID or command as being allowed to use\n    assistive access.\n\n    app_id\n        The bundle ID or command to install for assistive access.\n\n    enabled\n        Sets enabled or disabled status. Default is ``True``.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' assistive.install /usr/bin/osascript\n        salt '*' assistive.install com.smileonmymac.textexpander\n    "
    with TccDB() as db:
        try:
            return db.install(app_id, enable=enable)
        except sqlite3.Error as exc:
            raise CommandExecutionError('Error installing app({}): {}'.format(app_id, exc))

def installed(app_id):
    if False:
        for i in range(10):
            print('nop')
    "\n    Check if a bundle ID or command is listed in assistive access.\n    This will not check to see if it's enabled.\n\n    app_id\n        The bundle ID or command to check installed status.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' assistive.installed /usr/bin/osascript\n        salt '*' assistive.installed com.smileonmymac.textexpander\n    "
    with TccDB() as db:
        try:
            return db.installed(app_id)
        except sqlite3.Error as exc:
            raise CommandExecutionError('Error checking if app({}) is installed: {}'.format(app_id, exc))

def enable_(app_id, enabled=True):
    if False:
        while True:
            i = 10
    "\n    Enable or disable an existing assistive access application.\n\n    app_id\n        The bundle ID or command to set assistive access status.\n\n    enabled\n        Sets enabled or disabled status. Default is ``True``.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' assistive.enable /usr/bin/osascript\n        salt '*' assistive.enable com.smileonmymac.textexpander enabled=False\n    "
    with TccDB() as db:
        try:
            if enabled:
                return db.enable(app_id)
            else:
                return db.disable(app_id)
        except sqlite3.Error as exc:
            raise CommandExecutionError('Error setting enable to {} on app({}): {}'.format(enabled, app_id, exc))

def enabled(app_id):
    if False:
        print('Hello World!')
    "\n    Check if a bundle ID or command is listed in assistive access and\n    enabled.\n\n    app_id\n        The bundle ID or command to retrieve assistive access status.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' assistive.enabled /usr/bin/osascript\n        salt '*' assistive.enabled com.smileonmymac.textexpander\n    "
    with TccDB() as db:
        try:
            return db.enabled(app_id)
        except sqlite3.Error as exc:
            raise CommandExecutionError('Error checking if app({}) is enabled: {}'.format(app_id, exc))

def remove(app_id):
    if False:
        print('Hello World!')
    "\n    Remove a bundle ID or command as being allowed to use assistive access.\n\n    app_id\n        The bundle ID or command to remove from assistive access list.\n\n    CLI Example:\n\n    .. code-block:: bash\n\n        salt '*' assistive.remove /usr/bin/osascript\n        salt '*' assistive.remove com.smileonmymac.textexpander\n    "
    with TccDB() as db:
        try:
            return db.remove(app_id)
        except sqlite3.Error as exc:
            raise CommandExecutionError('Error removing app({}): {}'.format(app_id, exc))

class TccDB:

    def __init__(self, path=None):
        if False:
            return 10
        if path is None:
            path = TCC_DB_PATH
        self.path = path
        self.connection = None
        self.ge_mojave_and_catalina = False
        self.ge_bigsur_and_later = False

    def _check_table_digest(self):
        if False:
            print('Hello World!')
        with self.connection as conn:
            cursor = conn.execute("SELECT sql FROM sqlite_master WHERE name='access' and type='table'")
            for row in cursor.fetchall():
                digest = hashlib.sha1(row['sql'].encode()).hexdigest()[:10]
                if digest in ('ecc443615f', '80a4bb6912'):
                    self.ge_mojave_and_catalina = True
                elif digest in ('3d1c2a0e97', 'cef70648de'):
                    self.ge_bigsur_and_later = True
                else:
                    raise CommandExecutionError("TCC Database structure unknown for digest '{}'".format(digest))

    def _get_client_type(self, app_id):
        if False:
            i = 10
            return i + 15
        if app_id[0] == '/':
            return 1
        return 0

    def installed(self, app_id):
        if False:
            return 10
        with self.connection as conn:
            cursor = conn.execute("SELECT * from access WHERE client=? and service='kTCCServiceAccessibility'", (app_id,))
            for row in cursor.fetchall():
                if row:
                    return True
        return False

    def install(self, app_id, enable=True):
        if False:
            return 10
        client_type = self._get_client_type(app_id)
        auth_value = 1 if enable else 0
        if self.ge_bigsur_and_later:
            with self.connection as conn:
                conn.execute("\n                    INSERT or REPLACE INTO access VALUES (\n                        'kTCCServiceAccessibility',\n                        ?,\n                        ?,\n                        ?,\n                        4,\n                        1,\n                        NULL,\n                        NULL,\n                        0,\n                        'UNUSED',\n                        NULL,\n                        0,\n                        0\n                    )\n                    ", (app_id, client_type, auth_value))
        elif self.ge_mojave_and_catalina:
            with self.connection as conn:
                conn.execute("\n                    INSERT or REPLACE INTO access VALUES(\n                        'kTCCServiceAccessibility',\n                        ?,\n                        ?,\n                        ?,\n                        1,\n                        NULL,\n                        NULL,\n                        NULL,\n                        'UNUSED',\n                        NULL,\n                        0,\n                        0\n                    )\n                    ", (app_id, client_type, auth_value))
        return True

    def enabled(self, app_id):
        if False:
            for i in range(10):
                print('nop')
        if self.ge_bigsur_and_later:
            column = 'auth_value'
        elif self.ge_mojave_and_catalina:
            column = 'allowed'
        with self.connection as conn:
            cursor = conn.execute("SELECT * from access WHERE client=? and service='kTCCServiceAccessibility'", (app_id,))
            for row in cursor.fetchall():
                if row[column]:
                    return True
        return False

    def enable(self, app_id):
        if False:
            while True:
                i = 10
        if not self.installed(app_id):
            return False
        if self.ge_bigsur_and_later:
            column = 'auth_value'
        elif self.ge_mojave_and_catalina:
            column = 'allowed'
        with self.connection as conn:
            conn.execute("UPDATE access SET {} = ? WHERE client=? AND service IS 'kTCCServiceAccessibility'".format(column), (1, app_id))
        return True

    def disable(self, app_id):
        if False:
            return 10
        if not self.installed(app_id):
            return False
        if self.ge_bigsur_and_later:
            column = 'auth_value'
        elif self.ge_mojave_and_catalina:
            column = 'allowed'
        with self.connection as conn:
            conn.execute("UPDATE access SET {} = ? WHERE client=? AND service IS 'kTCCServiceAccessibility'".format(column), (0, app_id))
        return True

    def remove(self, app_id):
        if False:
            i = 10
            return i + 15
        if not self.installed(app_id):
            return False
        with self.connection as conn:
            conn.execute("DELETE from access where client IS ? AND service IS 'kTCCServiceAccessibility'", (app_id,))
        return True

    def __enter__(self):
        if False:
            i = 10
            return i + 15
        self.connection = sqlite3.connect(self.path)
        self.connection.row_factory = sqlite3.Row
        self._check_table_digest()
        return self

    def __exit__(self, *_):
        if False:
            print('Hello World!')
        self.connection.close()