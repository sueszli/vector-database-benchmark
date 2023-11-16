import os
import json
import unittest
import jc.parsers.lsblk
THIS_DIR = os.path.dirname(os.path.abspath(__file__))

class MyTests(unittest.TestCase):
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/centos-7.7/lsblk.out'), 'r', encoding='utf-8') as f:
        centos_7_7_lsblk = f.read()
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/ubuntu-18.04/lsblk.out'), 'r', encoding='utf-8') as f:
        ubuntu_18_4_lsblk = f.read()
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/centos-7.7/lsblk-allcols.out'), 'r', encoding='utf-8') as f:
        centos_7_7_lsblk_allcols = f.read()
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/ubuntu-18.04/lsblk-allcols.out'), 'r', encoding='utf-8') as f:
        ubuntu_18_4_lsblk_allcols = f.read()
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/centos-7.7/lsblk.json'), 'r', encoding='utf-8') as f:
        centos_7_7_lsblk_json = json.loads(f.read())
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/ubuntu-18.04/lsblk.json'), 'r', encoding='utf-8') as f:
        ubuntu_18_4_lsblk_json = json.loads(f.read())
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/centos-7.7/lsblk-allcols.json'), 'r', encoding='utf-8') as f:
        centos_7_7_lsblk_allcols_json = json.loads(f.read())
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/ubuntu-18.04/lsblk-allcols.json'), 'r', encoding='utf-8') as f:
        ubuntu_18_4_lsblk_allcols_json = json.loads(f.read())

    def test_lsblk_nodata(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Test 'lsblk' with no data\n        "
        self.assertEqual(jc.parsers.lsblk.parse('', quiet=True), [])

    def test_lsblk_centos_7_7(self):
        if False:
            print('Hello World!')
        "\n        Test 'lsblk' on Centos 7.7\n        "
        self.assertEqual(jc.parsers.lsblk.parse(self.centos_7_7_lsblk, quiet=True), self.centos_7_7_lsblk_json)

    def test_lsblk_ubuntu_18_4(self):
        if False:
            print('Hello World!')
        "\n        Test 'lsblk' on Ubuntu 18.4\n        "
        self.assertEqual(jc.parsers.lsblk.parse(self.ubuntu_18_4_lsblk, quiet=True), self.ubuntu_18_4_lsblk_json)

    def test_lsblk_allcols_centos_7_7(self):
        if False:
            print('Hello World!')
        "\n        Test 'lsblk -o +KNAME,FSTYPE,LABEL,UUID,PARTLABEL,PARTUUID,RA,MODEL,SERIAL,STATE,OWNER,GROUP,MODE,ALIGNMENT,MIN-IO,OPT-IO,PHY-SEC,LOG-SEC,ROTA,SCHED,RQ-SIZE,DISC-ALN,DISC-GRAN,DISC-MAX,DISC-ZERO,WSAME,WWN,RAND,PKNAME,HCTL,TRAN,REV,VENDOR' on Centos 7.7\n        "
        self.assertEqual(jc.parsers.lsblk.parse(self.centos_7_7_lsblk_allcols, quiet=True), self.centos_7_7_lsblk_allcols_json)

    def test_lsblk_allcols_ubuntu_18_4(self):
        if False:
            while True:
                i = 10
        "\n        Test 'lsblk -o +KNAME,FSTYPE,LABEL,UUID,PARTLABEL,PARTUUID,RA,MODEL,SERIAL,STATE,OWNER,GROUP,MODE,ALIGNMENT,MIN-IO,OPT-IO,PHY-SEC,LOG-SEC,ROTA,SCHED,RQ-SIZE,DISC-ALN,DISC-GRAN,DISC-MAX,DISC-ZERO,WSAME,WWN,RAND,PKNAME,HCTL,TRAN,REV,VENDOR' on Ubuntu 18.4\n        "
        self.assertEqual(jc.parsers.lsblk.parse(self.ubuntu_18_4_lsblk_allcols, quiet=True), self.ubuntu_18_4_lsblk_allcols_json)
if __name__ == '__main__':
    unittest.main()