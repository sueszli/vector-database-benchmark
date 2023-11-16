import os
import unittest
import json
from typing import Dict
from jc.parsers.os_prober import parse
THIS_DIR = os.path.dirname(os.path.abspath(__file__))

class MyTests(unittest.TestCase):

    def test_os_prober_nodata(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Test 'os_prober' with no data\n        "
        self.assertEqual(parse('', quiet=True), {})

    def test_os_prober_1(self):
        if False:
            return 10
        "\n        Test 'os_prober' 1\n        "
        self.assertEqual(parse('/dev/sda1:Windows 7 (loader):Windows:chain', quiet=True), {'partition': '/dev/sda1', 'name': 'Windows 7 (loader)', 'short_name': 'Windows', 'type': 'chain'})

    def test_os_prober_2(self):
        if False:
            print('Hello World!')
        "\n        Test 'os_prober' 2\n        "
        self.assertEqual(parse('/dev/sda1:Windows 10:Windows:chain', quiet=True), {'partition': '/dev/sda1', 'name': 'Windows 10', 'short_name': 'Windows', 'type': 'chain'})

    def test_os_prober_3(self):
        if False:
            print('Hello World!')
        "\n        Test 'os_prober' 3\n        "
        self.assertEqual(parse('/dev/sda1@/efi/Microsoft/Boot/bootmgfw.efi:Windows Boot Manager:Windows:efi', quiet=True), {'partition': '/dev/sda1', 'efi_bootmgr': '/efi/Microsoft/Boot/bootmgfw.efi', 'name': 'Windows Boot Manager', 'short_name': 'Windows', 'type': 'efi'})

    def test_os_prober_3_raw(self):
        if False:
            i = 10
            return i + 15
        "\n        Test 'os_prober' 3 with raw output\n        "
        self.assertEqual(parse('/dev/sda1@/efi/Microsoft/Boot/bootmgfw.efi:Windows Boot Manager:Windows:efi', quiet=True, raw=True), {'partition': '/dev/sda1@/efi/Microsoft/Boot/bootmgfw.efi', 'name': 'Windows Boot Manager', 'short_name': 'Windows', 'type': 'efi'})
if __name__ == '__main__':
    unittest.main()