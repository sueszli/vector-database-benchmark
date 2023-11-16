import os
import json
import unittest
import jc.parsers.lsusb
from jc.exceptions import ParseError
THIS_DIR = os.path.dirname(os.path.abspath(__file__))

class MyTests(unittest.TestCase):
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/centos-7.7/lsusb.out'), 'r', encoding='utf-8') as f:
        centos_7_7_lsusb = f.read()
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/centos-7.7/lsusb-v.out'), 'r', encoding='utf-8') as f:
        centos_7_7_lsusb_v = f.read()
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/centos-7.7/lsusb-v-single.out'), 'r', encoding='utf-8') as f:
        centos_7_7_lsusb_v_single = f.read()
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/generic/lsusb-test-attributes.out'), 'r', encoding='utf-8') as f:
        generic_lsusb_test_attributes = f.read()
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/generic/lsusb-test-attributes2.out'), 'r', encoding='utf-8') as f:
        generic_lsusb_test_attributes2 = f.read()
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/generic/lsusb-t.out'), 'r', encoding='utf-8') as f:
        generic_lsusb_t = f.read()
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/generic/lsusb-device-qualifier.out'), 'r', encoding='utf-8') as f:
        generic_lsusb_device_qualifier = f.read()
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/generic/lsusb-binary-object-store.out'), 'r', encoding='utf-8') as f:
        generic_lsusb_binary_object_store = f.read()
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/generic/lsusb-extra-hub-port-status-info.out'), 'r', encoding='utf-8') as f:
        generic_lsusb_extra_hub_port_status_info = f.read()
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/generic/lsusb-cdc-mbim.out'), 'r', encoding='utf-8') as f:
        generic_lsusb_cdc_mbim = f.read()
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/centos-7.7/lsusb.json'), 'r', encoding='utf-8') as f:
        centos_7_7_lsusb_json = json.loads(f.read())
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/centos-7.7/lsusb-v.json'), 'r', encoding='utf-8') as f:
        centos_7_7_lsusb_v_json = json.loads(f.read())
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/centos-7.7/lsusb-v-single.json'), 'r', encoding='utf-8') as f:
        centos_7_7_lsusb_v_single_json = json.loads(f.read())
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/generic/lsusb-test-attributes.json'), 'r', encoding='utf-8') as f:
        generic_lsusb_test_attributes_json = json.loads(f.read())
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/generic/lsusb-test-attributes2.json'), 'r', encoding='utf-8') as f:
        generic_lsusb_test_attributes2_json = json.loads(f.read())
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/generic/lsusb-device-qualifier.json'), 'r', encoding='utf-8') as f:
        generic_lsusb_devicez_qualifier_json = json.loads(f.read())
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/generic/lsusb-binary-object-store.json'), 'r', encoding='utf-8') as f:
        generic_lsusb_binary_object_store_json = json.loads(f.read())
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/generic/lsusb-extra-hub-port-status-info.json'), 'r', encoding='utf-8') as f:
        generic_lsusb_extra_hub_port_status_info_json = json.loads(f.read())
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/generic/lsusb-cdc-mbim.json'), 'r', encoding='utf-8') as f:
        generic_lsusb_cdc_mbim_json = json.loads(f.read())

    def test_lsusb_nodata(self):
        if False:
            while True:
                i = 10
        "\n        Test 'lsusb' with no data\n        "
        self.assertEqual(jc.parsers.lsusb.parse('', quiet=True), [])

    def test_lsusb_parse_error_generic(self):
        if False:
            i = 10
            return i + 15
        "\n        Test 'lsusb' with -t option (should raise ParseError)\n        "
        self.assertRaises(ParseError, jc.parsers.lsusb.parse, self.generic_lsusb_t, quiet=True)

    def test_lsusb_centos_7_7(self):
        if False:
            print('Hello World!')
        "\n        Test 'lsusb' on Centos 7.7\n        "
        self.assertEqual(jc.parsers.lsusb.parse(self.centos_7_7_lsusb, quiet=True), self.centos_7_7_lsusb_json)

    def test_lsusb_v_centos_7_7(self):
        if False:
            i = 10
            return i + 15
        "\n        Test 'lsusb -v' on Centos 7.7\n        "
        self.assertEqual(jc.parsers.lsusb.parse(self.centos_7_7_lsusb_v, quiet=True), self.centos_7_7_lsusb_v_json)

    def test_lsusb_v_single_centos_7_7(self):
        if False:
            i = 10
            return i + 15
        "\n        Test 'lsusb -v' with different hardware\n        "
        self.assertEqual(jc.parsers.lsusb.parse(self.centos_7_7_lsusb_v_single, quiet=True), self.centos_7_7_lsusb_v_single_json)

    def test_lsusb_test_attributes_generic(self):
        if False:
            while True:
                i = 10
        "\n        Test 'lsusb -v' with stress test attributes\n        "
        self.assertEqual(jc.parsers.lsusb.parse(self.generic_lsusb_test_attributes, quiet=True), self.generic_lsusb_test_attributes_json)

    def test_lsusb_test_attributes2_generic(self):
        if False:
            i = 10
            return i + 15
        "\n        Test 'lsusb -v' with stress test attributes 2\n        "
        self.assertEqual(jc.parsers.lsusb.parse(self.generic_lsusb_test_attributes2, quiet=True), self.generic_lsusb_test_attributes2_json)

    def test_lsusb_device_qualifier(self):
        if False:
            print('Hello World!')
        "\n        Test 'lsusb -v' with device qualifier section\n        "
        self.assertEqual(jc.parsers.lsusb.parse(self.generic_lsusb_device_qualifier, quiet=True), self.generic_lsusb_devicez_qualifier_json)

    def test_lsusb_binary_object_store(self):
        if False:
            print('Hello World!')
        "\n        Test 'lsusb -v' with binary object store section\n        "
        self.assertEqual(jc.parsers.lsusb.parse(self.generic_lsusb_binary_object_store, quiet=True), self.generic_lsusb_binary_object_store_json)

    def test_lsusb_extra_hub_port_status_info(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Test 'lsusb -v' with extra information in the hub port status section\n        "
        self.assertEqual(jc.parsers.lsusb.parse(self.generic_lsusb_extra_hub_port_status_info, quiet=True), self.generic_lsusb_extra_hub_port_status_info_json)

    def test_lsusb_cdc_mbim(self):
        if False:
            print('Hello World!')
        "\n        Test 'lsusb -v' with CDC MBIM and CDC MBIM Extended fields\n        "
        self.assertEqual(jc.parsers.lsusb.parse(self.generic_lsusb_cdc_mbim, quiet=True), self.generic_lsusb_cdc_mbim_json)
if __name__ == '__main__':
    unittest.main()