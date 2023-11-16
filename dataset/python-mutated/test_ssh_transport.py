from bzrlib.tests import TestCase
from bzrlib.errors import SSHVendorNotFound, UnknownSSH
from bzrlib.transport.ssh import OpenSSHSubprocessVendor, PLinkSubprocessVendor, SSHCorpSubprocessVendor, LSHSubprocessVendor, SSHVendorManager

class TestSSHVendorManager(SSHVendorManager):
    _ssh_version_string = ''

    def set_ssh_version_string(self, version):
        if False:
            i = 10
            return i + 15
        self._ssh_version_string = version

    def _get_ssh_version_string(self, args):
        if False:
            i = 10
            return i + 15
        return self._ssh_version_string

class SSHVendorManagerTests(TestCase):

    def test_register_vendor(self):
        if False:
            for i in range(10):
                print('nop')
        manager = TestSSHVendorManager()
        self.assertRaises(SSHVendorNotFound, manager.get_vendor, {})
        vendor = object()
        manager.register_vendor('vendor', vendor)
        self.assertIs(manager.get_vendor({'BZR_SSH': 'vendor'}), vendor)

    def test_default_vendor(self):
        if False:
            return 10
        manager = TestSSHVendorManager()
        self.assertRaises(SSHVendorNotFound, manager.get_vendor, {})
        vendor = object()
        manager.register_default_vendor(vendor)
        self.assertIs(manager.get_vendor({}), vendor)

    def test_get_vendor_by_environment(self):
        if False:
            i = 10
            return i + 15
        manager = TestSSHVendorManager()
        self.assertRaises(SSHVendorNotFound, manager.get_vendor, {})
        self.assertRaises(UnknownSSH, manager.get_vendor, {'BZR_SSH': 'vendor'})
        vendor = object()
        manager.register_vendor('vendor', vendor)
        self.assertIs(manager.get_vendor({'BZR_SSH': 'vendor'}), vendor)

    def test_get_vendor_by_inspection_openssh(self):
        if False:
            for i in range(10):
                print('nop')
        manager = TestSSHVendorManager()
        self.assertRaises(SSHVendorNotFound, manager.get_vendor, {})
        manager.set_ssh_version_string('OpenSSH')
        self.assertIsInstance(manager.get_vendor({}), OpenSSHSubprocessVendor)

    def test_get_vendor_by_inspection_sshcorp(self):
        if False:
            while True:
                i = 10
        manager = TestSSHVendorManager()
        self.assertRaises(SSHVendorNotFound, manager.get_vendor, {})
        manager.set_ssh_version_string('SSH Secure Shell')
        self.assertIsInstance(manager.get_vendor({}), SSHCorpSubprocessVendor)

    def test_get_vendor_by_inspection_lsh(self):
        if False:
            i = 10
            return i + 15
        manager = TestSSHVendorManager()
        self.assertRaises(SSHVendorNotFound, manager.get_vendor, {})
        manager.set_ssh_version_string('lsh')
        self.assertIsInstance(manager.get_vendor({}), LSHSubprocessVendor)

    def test_get_vendor_by_inspection_plink(self):
        if False:
            return 10
        manager = TestSSHVendorManager()
        self.assertRaises(SSHVendorNotFound, manager.get_vendor, {})
        manager.set_ssh_version_string('plink')
        self.assertRaises(SSHVendorNotFound, manager.get_vendor, {})

    def test_cached_vendor(self):
        if False:
            i = 10
            return i + 15
        manager = TestSSHVendorManager()
        self.assertRaises(SSHVendorNotFound, manager.get_vendor, {})
        vendor = object()
        manager.register_vendor('vendor', vendor)
        self.assertRaises(SSHVendorNotFound, manager.get_vendor, {})
        self.assertIs(manager.get_vendor({'BZR_SSH': 'vendor'}), vendor)
        self.assertIs(manager.get_vendor({}), vendor)
        manager.clear_cache()
        self.assertRaises(SSHVendorNotFound, manager.get_vendor, {})

    def test_get_vendor_search_order(self):
        if False:
            i = 10
            return i + 15
        manager = TestSSHVendorManager()
        self.assertRaises(SSHVendorNotFound, manager.get_vendor, {})
        default_vendor = object()
        manager.register_default_vendor(default_vendor)
        self.assertIs(manager.get_vendor({}), default_vendor)
        manager.clear_cache()
        manager.set_ssh_version_string('OpenSSH')
        self.assertIsInstance(manager.get_vendor({}), OpenSSHSubprocessVendor)
        manager.clear_cache()
        vendor = object()
        manager.register_vendor('vendor', vendor)
        self.assertIs(manager.get_vendor({'BZR_SSH': 'vendor'}), vendor)
        self.assertIs(manager.get_vendor({}), vendor)

    def test_get_vendor_from_path_win32_plink(self):
        if False:
            return 10
        manager = TestSSHVendorManager()
        manager.set_ssh_version_string('plink: Release 0.60')
        plink_path = 'C:/Program Files/PuTTY/plink.exe'
        vendor = manager.get_vendor({'BZR_SSH': plink_path})
        self.assertIsInstance(vendor, PLinkSubprocessVendor)
        args = vendor._get_vendor_specific_argv('user', 'host', 22, ['bzr'])
        self.assertEqual(args[0], plink_path)

    def test_get_vendor_from_path_nix_openssh(self):
        if False:
            i = 10
            return i + 15
        manager = TestSSHVendorManager()
        manager.set_ssh_version_string('OpenSSH_5.1p1 Debian-5, OpenSSL, 0.9.8g 19 Oct 2007')
        openssh_path = '/usr/bin/ssh'
        vendor = manager.get_vendor({'BZR_SSH': openssh_path})
        self.assertIsInstance(vendor, OpenSSHSubprocessVendor)
        args = vendor._get_vendor_specific_argv('user', 'host', 22, ['bzr'])
        self.assertEqual(args[0], openssh_path)

class SubprocessVendorsTests(TestCase):

    def test_openssh_command_arguments(self):
        if False:
            while True:
                i = 10
        vendor = OpenSSHSubprocessVendor()
        self.assertEqual(vendor._get_vendor_specific_argv('user', 'host', 100, command=['bzr']), ['ssh', '-oForwardX11=no', '-oForwardAgent=no', '-oClearAllForwardings=yes', '-oNoHostAuthenticationForLocalhost=yes', '-p', '100', '-l', 'user', 'host', 'bzr'])

    def test_openssh_subsystem_arguments(self):
        if False:
            while True:
                i = 10
        vendor = OpenSSHSubprocessVendor()
        self.assertEqual(vendor._get_vendor_specific_argv('user', 'host', 100, subsystem='sftp'), ['ssh', '-oForwardX11=no', '-oForwardAgent=no', '-oClearAllForwardings=yes', '-oNoHostAuthenticationForLocalhost=yes', '-p', '100', '-l', 'user', '-s', 'host', 'sftp'])

    def test_sshcorp_command_arguments(self):
        if False:
            while True:
                i = 10
        vendor = SSHCorpSubprocessVendor()
        self.assertEqual(vendor._get_vendor_specific_argv('user', 'host', 100, command=['bzr']), ['ssh', '-x', '-p', '100', '-l', 'user', 'host', 'bzr'])

    def test_sshcorp_subsystem_arguments(self):
        if False:
            while True:
                i = 10
        vendor = SSHCorpSubprocessVendor()
        self.assertEqual(vendor._get_vendor_specific_argv('user', 'host', 100, subsystem='sftp'), ['ssh', '-x', '-p', '100', '-l', 'user', '-s', 'sftp', 'host'])

    def test_lsh_command_arguments(self):
        if False:
            while True:
                i = 10
        vendor = LSHSubprocessVendor()
        self.assertEqual(vendor._get_vendor_specific_argv('user', 'host', 100, command=['bzr']), ['lsh', '-p', '100', '-l', 'user', 'host', 'bzr'])

    def test_lsh_subsystem_arguments(self):
        if False:
            for i in range(10):
                print('nop')
        vendor = LSHSubprocessVendor()
        self.assertEqual(vendor._get_vendor_specific_argv('user', 'host', 100, subsystem='sftp'), ['lsh', '-p', '100', '-l', 'user', '--subsystem', 'sftp', 'host'])

    def test_plink_command_arguments(self):
        if False:
            i = 10
            return i + 15
        vendor = PLinkSubprocessVendor()
        self.assertEqual(vendor._get_vendor_specific_argv('user', 'host', 100, command=['bzr']), ['plink', '-x', '-a', '-ssh', '-2', '-batch', '-P', '100', '-l', 'user', 'host', 'bzr'])

    def test_plink_subsystem_arguments(self):
        if False:
            for i in range(10):
                print('nop')
        vendor = PLinkSubprocessVendor()
        self.assertEqual(vendor._get_vendor_specific_argv('user', 'host', 100, subsystem='sftp'), ['plink', '-x', '-a', '-ssh', '-2', '-batch', '-P', '100', '-l', 'user', '-s', 'host', 'sftp'])