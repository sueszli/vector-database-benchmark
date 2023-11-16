import unittest
import os
from pocsuite3.api import OSShellcodes
from pocsuite3.lib.core.data import paths
from pocsuite3.lib.core.enums import SHELLCODE_CONNECTION, OS, OS_ARCH

class TestCase(unittest.TestCase):

    def setUp(self):
        if False:
            return 10
        self.connect_back_ip = '127.0.0.1'
        self.bad_chars = ['\x00', '\n', '\r', ';']
        self.shellpath = os.path.join(paths.POCSUITE_TMP_PATH, 'payload.jar')

    def tearDown(self):
        if False:
            i = 10
            return i + 15
        if os.path.exists(self.shellpath):
            os.unlink(self.shellpath)

    def test_win_x86_bind(self):
        if False:
            print('Hello World!')
        os_target = OS.WINDOWS
        os_target_arch = OS_ARCH.X86
        dll_funcs = ['pcap_findalldevs', 'pcap_close', 'pcap_compile', 'pcap_datalink', 'pcap_datalink_val_to_description', 'pcap_dump', 'pcap_dump_close', 'pcap_dump_open', 'pcap_file', 'pcap_freecode', 'pcap_geterr', 'pcap_getevent', 'pcap_lib_version', 'pcap_lookupdev', 'pcap_lookupnet', 'pcap_loop', 'pcap_open_live', 'pcap_open_offline', 'pcap_setfilter', 'pcap_snapshot', 'pcap_stats']
        s = OSShellcodes(os_target, os_target_arch, self.connect_back_ip, 6666, self.bad_chars)
        connection_type = SHELLCODE_CONNECTION.BIND
        filename = 'osshell_x86_bind'
        filepath = os.path.join(paths.POCSUITE_TMP_PATH, filename) + '.exe'
        shellcode = s.create_shellcode(connection_type, encode='', make_exe=1, debug=0, filename=filename)
        self.assertTrue(os.path.exists(filepath))
        os.unlink(filepath)

    def test_win_x86_reverse(self):
        if False:
            return 10
        os_target = OS.WINDOWS
        os_target_arch = OS_ARCH.X86
        dll_funcs = ['pcap_findalldevs', 'pcap_close', 'pcap_compile', 'pcap_datalink', 'pcap_datalink_val_to_description', 'pcap_dump', 'pcap_dump_close', 'pcap_dump_open', 'pcap_file', 'pcap_freecode', 'pcap_geterr', 'pcap_getevent', 'pcap_lib_version', 'pcap_lookupdev', 'pcap_lookupnet', 'pcap_loop', 'pcap_open_live', 'pcap_open_offline', 'pcap_setfilter', 'pcap_snapshot', 'pcap_stats']
        s = OSShellcodes(os_target, os_target_arch, self.connect_back_ip, 6666, self.bad_chars)
        connection_type = SHELLCODE_CONNECTION.REVERSE
        filename = 'osshell_x86_reverse'
        filepath = os.path.join(paths.POCSUITE_TMP_PATH, filename) + '.exe'
        shellcode = s.create_shellcode(connection_type, encode='', make_exe=1, debug=0, filename=filename)
        self.assertTrue(os.path.exists(filepath))
        os.unlink(filepath)

    def test_win_x64_bind(self):
        if False:
            while True:
                i = 10
        pass

    def test_win_x64_reverse(self):
        if False:
            while True:
                i = 10
        pass

    def test_linux_x86_bind(self):
        if False:
            i = 10
            return i + 15
        pass

    def test_linux_x86_reverse(self):
        if False:
            while True:
                i = 10
        pass

    def test_linux_x64_bind(self):
        if False:
            return 10
        pass

    def test_linux_x64_reverse(self):
        if False:
            while True:
                i = 10
        pass