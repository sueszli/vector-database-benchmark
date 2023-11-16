import unittest
from time import sleep
from os.path import exists, join
from os import mkdir, rmdir, remove
from multiprocessing import Process
from six import PY2, StringIO, BytesIO, b, assertRaisesRegex, assertCountEqual
from impacket.smb import SMB_DIALECT
from impacket.smbserver import normalize_path, isInFileJail, SimpleSMBServer
from impacket.smbconnection import SMBConnection, SessionError, compute_lmhash, compute_nthash

class SMBServerUnitTests(unittest.TestCase):
    """Unit tests for the SMBServer
    """

    def test_normalize_path(self):
        if False:
            while True:
                i = 10
        'Test file path normalization.\n        '
        self.assertEqual(normalize_path('filepath'), 'filepath')
        self.assertEqual(normalize_path('filepath\\'), 'filepath')
        self.assertEqual(normalize_path('filepath\\\\'), 'filepath')
        self.assertEqual(normalize_path('\\filepath\\'), 'filepath')
        self.assertEqual(normalize_path('\\\\filepath\\'), '/filepath')
        self.assertEqual(normalize_path('.\\filepath'), 'filepath')
        self.assertEqual(normalize_path('.\\.\\filepath'), 'filepath')
        self.assertEqual(normalize_path('..\\.\\filepath'), '../filepath')
        self.assertEqual(normalize_path('..\\filepath\\..\\..\\filepath'), '../../filepath')
        self.assertEqual(normalize_path('/filepath'), 'filepath')
        self.assertEqual(normalize_path('//filepath'), '/filepath')
        self.assertEqual(normalize_path('./filepath'), 'filepath')
        self.assertEqual(normalize_path('././filepath'), 'filepath')
        self.assertEqual(normalize_path('.././filepath'), '../filepath')
        self.assertEqual(normalize_path('../filepath/../../filepath'), '../../filepath')
        self.assertEqual(normalize_path('filepath', ''), 'filepath')
        self.assertEqual(normalize_path('/filepath', ''), '/filepath')
        self.assertEqual(normalize_path('//filepath', ''), '//filepath')
        self.assertEqual(normalize_path('filepath', 'path'), 'filepath')
        self.assertEqual(normalize_path('/filepath', 'path'), 'filepath')
        self.assertEqual(normalize_path('//filepath', 'path'), '/filepath')

    def test_isInFileJail(self):
        if False:
            i = 10
            return i + 15
        'Test validation of common prefix path.\n        '
        jail_path = '/tmp/jail_path'
        self.assertTrue(isInFileJail(jail_path, 'filename'))
        self.assertTrue(isInFileJail(jail_path, './filename'))
        self.assertTrue(isInFileJail(jail_path, '../jail_path/filename'))
        self.assertFalse(isInFileJail(jail_path, '/filename'))
        self.assertFalse(isInFileJail(jail_path, '/tmp/filename'))
        self.assertFalse(isInFileJail(jail_path, '../filename'))
        self.assertFalse(isInFileJail(jail_path, '../../filename'))
        jail_path = ''
        self.assertTrue(isInFileJail(jail_path, 'filename'))
        self.assertTrue(isInFileJail(jail_path, './filename'))
        self.assertFalse(isInFileJail(jail_path, '../jail_path/filename'))
        self.assertFalse(isInFileJail(jail_path, '/filename'))
        self.assertFalse(isInFileJail(jail_path, '/tmp/filename'))
        self.assertFalse(isInFileJail(jail_path, '../filename'))
        self.assertFalse(isInFileJail(jail_path, '../../filename'))

class SimpleSMBServerFuncTests(unittest.TestCase):
    """Pseudo functional tests for the SimpleSMBServer.

    These are pseudo functional as we're using our own SMBConnection classes. For a complete functional test
    we should (and can) use for example Samba's smbclient or similar.
    """
    server = None
    server_smb2_support = False
    client_preferred_dialect = None
    address = '127.0.0.1'
    port = 1445
    username = 'UserName'
    password = 'Password'
    domain = 'DOMAIN'
    lmhash = compute_lmhash(password)
    nthash = compute_nthash(password)
    unicode_share_file = 'test\u202etest'
    unicode_username = 'User\u202eName'
    share_name = 'share'
    share_path = 'jail_dir'
    share_file = 'jail_file'
    share_new_file = 'jail_new_file'
    share_unjailed_file = 'unjailed_file'
    share_unjailed_new_file = 'unjailed_new_file'
    share_new_content = 'some content'
    share_directory = 'directory'
    share_new_directory = 'new_directory'
    share_unjailed_directory = 'unjailed_directory'
    share_unjailed_new_directory = 'unjailed_new_directory'
    share_list = ['.', '..', share_file, share_directory, unicode_share_file]

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        'Creates folders and files required for testing the list, put and get functionality.\n        '
        self.server_process = None
        for d in [self.share_path, self.share_unjailed_directory, join(self.share_path, self.share_directory)]:
            if not exists(d):
                mkdir(d)
        for f in [self.share_unjailed_file, join(self.share_path, self.share_file), join(self.share_path, self.unicode_share_file)]:
            if not exists(f):
                with open(f, 'a') as fd:
                    fd.write(self.share_new_content)

    def tearDown(self):
        if False:
            print('Hello World!')
        'Removes folders and files used for testing.\n        '
        for f in [self.share_unjailed_file, self.share_unjailed_new_file, join(self.share_path, self.share_file), join(self.share_path, self.unicode_share_file), join(self.share_path, self.share_new_file)]:
            if exists(f):
                remove(f)
        for d in [self.share_unjailed_directory, self.share_unjailed_new_directory, join(self.share_path, self.share_directory), join(self.share_path, self.share_new_directory), self.share_path]:
            if exists(d):
                rmdir(d)
        self.stop_smbserver()

    def get_smbserver(self, add_credential=True, add_share=True):
        if False:
            print('Hello World!')
        smbserver = SimpleSMBServer(listenAddress=self.address, listenPort=int(self.port))
        if add_credential:
            smbserver.addCredential(self.username, 0, self.lmhash, self.nthash)
        if add_share:
            smbserver.addShare(self.share_name, self.share_path)
        if self.server_smb2_support is not None:
            smbserver.setSMB2Support(self.server_smb2_support)
        return smbserver

    def get_smbclient(self):
        if False:
            return 10
        smbclient = SMBConnection(self.address, self.address, sess_port=int(self.port), preferredDialect=self.client_preferred_dialect)
        return smbclient

    def start_smbserver(self, server):
        if False:
            print('Hello World!')
        'Starts the SimpleSMBServer process.\n        '
        self.server = server
        self.server_process = Process(target=server.start)
        self.server_process.start()

    def stop_smbserver(self):
        if False:
            i = 10
            return i + 15
        'Stops the SimpleSMBServer process and wait for insider threads to join.\n        '
        if self.server:
            self.server.stop()
            self.server = None
        if self.server_process:
            self.server_process.terminate()
            sleep(0.1)
            self.server_process = None

    def test_smbserver_login_valid(self):
        if False:
            while True:
                i = 10
        'Test authentication using valid password and LM/NTHash.\n        '
        server = self.get_smbserver(add_share=False)
        self.start_smbserver(server)
        client = self.get_smbclient()
        client.login(self.username, self.password)
        client.close()
        client = self.get_smbclient()
        client.login(self.username, '', lmhash=self.lmhash, nthash=self.nthash)
        client.close()

    def test_smbserver_login_invalid(self):
        if False:
            return 10
        'Test authentication using invalid password and LM/NTHash.\n        '
        server = self.get_smbserver(add_share=False)
        self.start_smbserver(server)
        client = self.get_smbclient()
        with assertRaisesRegex(self, SessionError, 'STATUS_LOGON_FAILURE'):
            client.login(self.username, 'SomeInvalidPassword')
        client.close()
        client = self.get_smbclient()
        with assertRaisesRegex(self, SessionError, 'STATUS_LOGON_FAILURE'):
            client.login('InvalidUser', '', lmhash=self.lmhash, nthash=self.nthash)
        client.close()
        client = self.get_smbclient()
        with assertRaisesRegex(self, SessionError, 'STATUS_LOGON_FAILURE'):
            client.login(self.username, '', lmhash=self.nthash, nthash=self.lmhash)
        client.close()

    def test_smbserver_unicode_login(self):
        if False:
            i = 10
            return i + 15
        'Test authentication using a unicode username.\n        '
        server = self.get_smbserver(add_credential=False, add_share=False)
        server.addCredential(self.unicode_username, 0, self.lmhash, self.nthash)
        self.start_smbserver(server)
        client = self.get_smbclient()
        client.login(self.unicode_username, self.password)
        client.close()

    def test_smbserver_list_shares(self):
        if False:
            print('Hello World!')
        'Test listing shares.\n        '
        server = self.get_smbserver()
        self.start_smbserver(server)
        client = self.get_smbclient()
        with assertRaisesRegex(self, SessionError, 'STATUS_ACCESS_DENIED'):
            client.listShares()
        client.login(self.username, self.password)
        shares = client.listShares()
        shares_names = [share['shi1_netname'][:-1] for share in shares]
        assertCountEqual(self, [self.share_name.upper(), 'IPC$'], shares_names)
        client.close()

    def test_smbserver_connect_disconnect_tree(self):
        if False:
            print('Hello World!')
        'Test connecting/disconnecting to a share tree.\n        '
        server = self.get_smbserver()
        self.start_smbserver(server)
        client = self.get_smbclient()
        with assertRaisesRegex(self, SessionError, 'STATUS_ACCESS_DENIED'):
            client.connectTree(self.share_name)
        client.login(self.username, self.password)
        tree_id = client.connectTree(self.share_name)
        client.disconnectTree(tree_id)
        with assertRaisesRegex(self, SessionError, 'STATUS_OBJECT_PATH_NOT_FOUND'):
            client.connectTree('unexistent')
        client.close()

    @unittest.skipIf(PY2, 'Unicode filename expected failing in Python 2.x')
    def test_smbserver_list_path(self):
        if False:
            print('Hello World!')
        'Test listing files in a shared folder.\n        '
        server = self.get_smbserver()
        self.start_smbserver(server)
        client = self.get_smbclient()
        with assertRaisesRegex(self, SessionError, 'STATUS_ACCESS_DENIED'):
            client.listPath(self.share_name, '/')
        client.login(self.username, self.password)
        files = client.listPath(self.share_name, self.share_file)
        assertCountEqual(self, [f.get_longname() for f in files], [self.share_file])
        files = client.listPath(self.share_name, self.share_directory)
        assertCountEqual(self, [f.get_longname() for f in files], [self.share_directory])
        files = client.listPath(self.share_name, self.unicode_share_file)
        assertCountEqual(self, [f.get_longname() for f in files], [self.unicode_share_file])
        files = client.listPath(self.share_name, '*')
        assertCountEqual(self, [f.get_longname() for f in files], self.share_list)
        with assertRaisesRegex(self, SessionError, 'STATUS_OBJECT_PATH_SYNTAX_BAD'):
            client.listPath(self.share_name, join('..', self.share_unjailed_file))
        with assertRaisesRegex(self, SessionError, 'STATUS_NO_SUCH_FILE'):
            client.listPath(self.share_name, 'unexistent')
        client.close()

    def test_smbserver_put(self):
        if False:
            return 10
        'Test writing files to a shared folder.\n        '
        server = self.get_smbserver()
        self.start_smbserver(server)
        client = self.get_smbclient()
        local_file = StringIO(self.share_new_content)
        with assertRaisesRegex(self, SessionError, 'STATUS_ACCESS_DENIED'):
            client.putFile(self.share_name, self.share_new_file, local_file.read)
        self.assertFalse(exists(join(self.share_path, self.share_new_file)))
        local_file = StringIO(self.share_new_content)
        client.login(self.username, self.password)
        client.putFile(self.share_name, self.share_new_file, local_file.read)
        self.assertTrue(exists(join(self.share_path, self.share_new_file)))
        with open(join(self.share_path, self.share_new_file), 'r') as fd:
            self.assertEqual(fd.read(), self.share_new_content)
        local_file = StringIO(self.share_new_content)
        with assertRaisesRegex(self, SessionError, 'STATUS_OBJECT_PATH_SYNTAX_BAD'):
            client.putFile(self.share_name, join('..', self.share_unjailed_new_file), local_file.read)
        self.assertFalse(exists(self.share_unjailed_new_file))
        client.close()

    def test_smbserver_get_file(self):
        if False:
            for i in range(10):
                print('nop')
        'Test reading files from a shared folder.\n        '
        server = self.get_smbserver()
        self.start_smbserver(server)
        client = self.get_smbclient()
        local_file = BytesIO()
        with assertRaisesRegex(self, SessionError, 'STATUS_ACCESS_DENIED'):
            client.getFile(self.share_name, self.share_file, local_file.write)
        local_file = BytesIO()
        client.login(self.username, self.password)
        client.getFile(self.share_name, self.share_file, local_file.write)
        local_file.seek(0)
        self.assertEqual(local_file.read(), b(self.share_new_content))
        local_file = BytesIO()
        with assertRaisesRegex(self, SessionError, 'STATUS_OBJECT_PATH_SYNTAX_BAD'):
            client.getFile(self.share_name, join('..', self.share_unjailed_file), local_file.write)
        local_file.seek(0)
        self.assertEqual(local_file.read(), b(''))
        with assertRaisesRegex(self, SessionError, 'STATUS_NO_SUCH_FILE'):
            client.getFile(self.share_name, 'unexistent', local_file.write)
        client.close()

    @unittest.skipIf(PY2, 'Unicode filename expected failing in Python 2.x')
    def test_smbserver_get_unicode_file(self):
        if False:
            while True:
                i = 10
        'Test reading unicode files from a shared folder.\n        '
        server = self.get_smbserver()
        self.start_smbserver(server)
        client = self.get_smbclient()
        local_file = BytesIO()
        client.login(self.username, self.password)
        client.getFile(self.share_name, self.unicode_share_file, local_file.write)
        local_file.seek(0)
        self.assertEqual(local_file.read(), b(self.share_new_content))
        client.close()

    def test_smbserver_delete_file(self):
        if False:
            i = 10
            return i + 15
        'Test deleting files from a shared folder.\n        '
        server = self.get_smbserver()
        self.start_smbserver(server)
        client = self.get_smbclient()
        with assertRaisesRegex(self, SessionError, 'STATUS_ACCESS_DENIED'):
            client.deleteFile(self.share_name, self.share_file)
        self.assertTrue(exists(join(self.share_path, self.share_file)))
        client.login(self.username, self.password)
        with assertRaisesRegex(self, SessionError, 'STATUS_OBJECT_PATH_SYNTAX_BAD'):
            client.deleteFile(self.share_name, join('..', self.share_unjailed_file))
        self.assertTrue(exists(self.share_unjailed_file))
        client.deleteFile(self.share_name, self.share_file)
        self.assertFalse(exists(join(self.share_path, self.share_file)))
        with assertRaisesRegex(self, SessionError, 'STATUS_NO_SUCH_FILE'):
            client.deleteFile(self.share_name, 'unexistent')
        client.close()

    def test_smbserver_create_directory(self):
        if False:
            i = 10
            return i + 15
        'Test creating a directory on a shared folder.\n        '
        server = self.get_smbserver()
        self.start_smbserver(server)
        client = self.get_smbclient()
        with assertRaisesRegex(self, SessionError, 'STATUS_ACCESS_DENIED'):
            client.createDirectory(self.share_name, self.share_new_directory)
        self.assertFalse(exists(join(self.share_path, self.share_new_directory)))
        client.login(self.username, self.password)
        client.createDirectory(self.share_name, self.share_new_directory)
        self.assertTrue(exists(join(self.share_path, self.share_new_directory)))
        with assertRaisesRegex(self, SessionError, 'STATUS_OBJECT_PATH_SYNTAX_BAD'):
            client.createDirectory(self.share_name, join('..', self.share_unjailed_new_directory))
        self.assertFalse(exists(self.share_unjailed_new_directory))
        client.close()

    def test_smbserver_rename_file(self):
        if False:
            i = 10
            return i + 15
        'Test renaming files in a shared folder.\n        '
        server = self.get_smbserver()
        self.start_smbserver(server)
        client = self.get_smbclient()
        with assertRaisesRegex(self, SessionError, 'STATUS_ACCESS_DENIED'):
            client.rename(self.share_name, self.share_file, self.share_new_file)
        self.assertTrue(exists(join(self.share_path, self.share_file)))
        self.assertFalse(exists(join(self.share_path, self.share_new_file)))
        client.login(self.username, self.password)
        with assertRaisesRegex(self, SessionError, 'STATUS_OBJECT_PATH_SYNTAX_BAD'):
            client.rename(self.share_name, self.share_file, join('..', self.share_unjailed_new_file))
        self.assertTrue(exists(join(self.share_path, self.share_file)))
        self.assertFalse(exists(self.share_unjailed_new_file))
        with assertRaisesRegex(self, SessionError, 'STATUS_OBJECT_PATH_SYNTAX_BAD'):
            client.rename(self.share_name, join('..', self.share_unjailed_file), self.share_new_file)
        self.assertTrue(exists(self.share_unjailed_file))
        self.assertFalse(exists(self.share_new_file))
        with assertRaisesRegex(self, SessionError, 'STATUS_OBJECT_PATH_SYNTAX_BAD'):
            client.rename(self.share_name, join('..', self.share_unjailed_file), join('..', self.share_unjailed_new_file))
        self.assertTrue(exists(self.share_unjailed_file))
        self.assertFalse(exists(self.share_unjailed_new_file))
        client.rename(self.share_name, self.share_file, self.share_new_file)
        self.assertFalse(exists(join(self.share_path, self.share_file)))
        self.assertTrue(exists(join(self.share_path, self.share_new_file)))
        with open(join(self.share_path, self.share_new_file), 'r') as fd:
            self.assertEqual(fd.read(), self.share_new_content)
        with assertRaisesRegex(self, SessionError, 'STATUS_NO_SUCH_FILE'):
            client.rename(self.share_name, 'unexistent', self.share_new_file)
        client.close()

    def test_smbserver_open_close_file(self):
        if False:
            return 10
        'Test opening and closing files in a shared folder.\n        '
        server = self.get_smbserver()
        self.start_smbserver(server)
        client = self.get_smbclient()
        client.login(self.username, self.password)
        tree_id = client.connectTree(self.share_name)
        file_id = client.openFile(tree_id, self.share_file)
        with assertRaisesRegex(self, SessionError, 'STATUS_OBJECT_PATH_SYNTAX_BAD'):
            client.openFile(tree_id, join('..', self.share_unjailed_file))
        with assertRaisesRegex(self, SessionError, 'STATUS_NO_SUCH_FILE'):
            client.openFile(tree_id, 'unexistent')
        with self.assertRaises(SessionError):
            client.closeFile(tree_id, 123)
        with self.assertRaises(SessionError):
            client.closeFile(123, file_id)
        with self.assertRaises(SessionError):
            client.closeFile('123', file_id)
        client.closeFile(tree_id, file_id)
        client.disconnectTree(tree_id)
        client.close()

    def test_smbserver_query_info_file(self):
        if False:
            while True:
                i = 10
        'Test query info on a file in a shared folder.\n        '
        server = self.get_smbserver()
        self.start_smbserver(server)
        client = self.get_smbclient()
        client.login(self.username, self.password)
        tree_id = client.connectTree(self.share_name)
        file_id = client.openFile(tree_id, self.share_file)
        file_info = client.queryInfo(tree_id, file_id)
        self.assertEqual(file_info['AllocationSize'], len(self.share_new_content))
        self.assertEqual(file_info['EndOfFile'], len(self.share_new_content))
        self.assertEqual(file_info['Directory'], 0)
        client.closeFile(tree_id, file_id)
        client.disconnectTree(tree_id)
        client.close()

    @unittest.skip('Query directory not implemented on client')
    def test_smbserver_query_info_directory(self):
        if False:
            return 10
        'Test query info on a directory in a shared folder.\n        '
        server = self.get_smbserver()
        self.start_smbserver(server)
        client = self.get_smbclient()
        client.login(self.username, self.password)
        tree_id = client.connectTree(self.share_name)
        directory_id = client.openFile(tree_id, self.share_directory)
        directory_info = client.queryInfo(tree_id, directory_id)
        self.assertEqual(directory_info['AllocationSize'], len(self.share_new_content))
        self.assertEqual(directory_info['EndOfFile'], len(self.share_new_content))
        self.assertEqual(directory_info['Directory'], 1)
        client.closeFile(tree_id, directory_id)
        client.disconnectTree(tree_id)
        client.close()

class SimpleSMBServer2FuncTestsClientFallBack(SimpleSMBServerFuncTests):
    server_smb2_support = True
    client_preferred_dialect = SMB_DIALECT

class SimpleSMBServer2FuncTests(SimpleSMBServerFuncTests):
    server_smb2_support = True
    share_list = [SimpleSMBServerFuncTests.share_file, SimpleSMBServerFuncTests.share_directory, SimpleSMBServerFuncTests.unicode_share_file]

    def test_smbserver_delete_directory(self):
        if False:
            for i in range(10):
                print('nop')
        'Test deleting directories from a shared folder.\n\n        This is only tested in SMB2 as SMB_COM_CHECK_DIRECTORY is not\n        implemented yet in SMB, the SMB2 client uses a query info instead.\n        '
        server = self.get_smbserver()
        self.start_smbserver(server)
        client = self.get_smbclient()
        with assertRaisesRegex(self, SessionError, 'STATUS_ACCESS_DENIED'):
            client.deleteDirectory(self.share_name, self.share_directory)
        self.assertTrue(exists(join(self.share_path, self.share_directory)))
        client.login(self.username, self.password)
        with assertRaisesRegex(self, SessionError, 'STATUS_OBJECT_PATH_SYNTAX_BAD'):
            client.deleteDirectory(self.share_name, join('..', self.share_unjailed_directory))
        client.deleteDirectory(self.share_name, self.share_directory)
        self.assertFalse(exists(join(self.share_path, self.share_directory)))
        with assertRaisesRegex(self, SessionError, 'STATUS_NO_SUCH_FILE'):
            client.deleteDirectory(self.share_name, 'unexistent')
        client.close()
if __name__ == '__main__':
    unittest.main(verbosity=1)