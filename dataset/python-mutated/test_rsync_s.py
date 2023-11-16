import os
import sys
import time
import json
import unittest
import jc.parsers.rsync_s
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if not sys.platform.startswith('win32'):
    os.environ['TZ'] = 'America/Los_Angeles'
    time.tzset()

class MyTests(unittest.TestCase):
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/centos-7.7/rsync-i.out'), 'r', encoding='utf-8') as f:
        centos_7_7_rsync_i = f.read()
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/generic/rsync-i.out'), 'r', encoding='utf-8') as f:
        generic_rsync_i = f.read()
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/centos-7.7/rsync-ivvv.out'), 'r', encoding='utf-8') as f:
        centos_7_7_rsync_ivvv = f.read()
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/osx-10.14.6/rsync-i-vvv.out'), 'r', encoding='utf-8') as f:
        osx_10_14_6_rsync_ivvv = f.read()
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/centos-7.7/rsync-ivvv-nochange.out'), 'r', encoding='utf-8') as f:
        centos_7_7_rsync_ivvv_nochange = f.read()
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/osx-10.14.6/rsync-i-vvv-nochange.out'), 'r', encoding='utf-8') as f:
        osx_10_14_6_rsync_ivvv_nochange = f.read()
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/centos-7.7/rsync-i-logfile.out'), 'r', encoding='utf-8') as f:
        centos_7_7_rsync_i_logfile = f.read()
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/centos-7.7/rsync-i-v-logfile.out'), 'r', encoding='utf-8') as f:
        centos_7_7_rsync_i_v_logfile = f.read()
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/centos-7.7/rsync-i-vv-logfile.out'), 'r', encoding='utf-8') as f:
        centos_7_7_rsync_i_vv_logfile = f.read()
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/centos-7.7/rsync-i-vvv-logfile.out'), 'r', encoding='utf-8') as f:
        centos_7_7_rsync_i_vvv_logfile = f.read()
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/centos-7.7/rsync-i-vvv-logfile-nochange.out'), 'r', encoding='utf-8') as f:
        centos_7_7_rsync_i_vvv_logfile_nochange = f.read()
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/osx-10.14.6/rsync-i-vvv-logfile-nochange.out'), 'r', encoding='utf-8') as f:
        osx_10_14_6_rsync_i_vvv_logfile_nochange = f.read()
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/centos-7.7/rsync-i-streaming.json'), 'r', encoding='utf-8') as f:
        centos_7_7_rsync_i_streaming_json = json.loads(f.read())
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/generic/rsync-i-streaming.json'), 'r', encoding='utf-8') as f:
        generic_rsync_i_streaming_json = json.loads(f.read())
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/centos-7.7/rsync-ivvv-streaming.json'), 'r', encoding='utf-8') as f:
        centos_7_7_rsync_ivvv_streaming_json = json.loads(f.read())
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/osx-10.14.6/rsync-i-vvv-streaming.json'), 'r', encoding='utf-8') as f:
        osx_10_14_6_rsync_ivvv_streaming_json = json.loads(f.read())
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/centos-7.7/rsync-ivvv-nochange-streaming.json'), 'r', encoding='utf-8') as f:
        centos_7_7_rsync_ivvv_nochange_streaming_json = json.loads(f.read())
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/osx-10.14.6/rsync-i-vvv-nochange-streaming.json'), 'r', encoding='utf-8') as f:
        osx_10_14_6_rsync_ivvv_nochange_streaming_json = json.loads(f.read())
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/centos-7.7/rsync-i-logfile-streaming.json'), 'r', encoding='utf-8') as f:
        centos_7_7_rsync_i_logfile_streaming_json = json.loads(f.read())
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/centos-7.7/rsync-i-v-logfile-streaming.json'), 'r', encoding='utf-8') as f:
        centos_7_7_rsync_i_v_logfile_streaming_json = json.loads(f.read())
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/centos-7.7/rsync-i-vv-logfile-streaming.json'), 'r', encoding='utf-8') as f:
        centos_7_7_rsync_i_vv_logfile_streaming_json = json.loads(f.read())
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/centos-7.7/rsync-i-vvv-logfile-streaming.json'), 'r', encoding='utf-8') as f:
        centos_7_7_rsync_i_vvv_logfile_streaming_json = json.loads(f.read())
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/centos-7.7/rsync-i-vvv-logfile-nochange-streaming.json'), 'r', encoding='utf-8') as f:
        centos_7_7_rsync_i_vvv_logfile_nochange_streaming_json = json.loads(f.read())
    with open(os.path.join(THIS_DIR, os.pardir, 'tests/fixtures/osx-10.14.6/rsync-i-vvv-logfile-nochange-streaming.json'), 'r', encoding='utf-8') as f:
        osx_10_14_6_rsync_i_vvv_logfile_nochange_streaming_json = json.loads(f.read())

    def test_rsync_s_nodata(self):
        if False:
            print('Hello World!')
        "\n        Test 'rsync' with no data\n        "
        self.assertEqual(list(jc.parsers.rsync_s.parse([], quiet=True)), [])

    def test_rsync_s_i_centos_7_7(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Test 'rsync -i' on Centos 7.7\n        "
        self.assertEqual(list(jc.parsers.rsync_s.parse(self.centos_7_7_rsync_i.splitlines(), quiet=True)), self.centos_7_7_rsync_i_streaming_json)

    def test_rsync_s_i_generic(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Test 'rsync -i'\n        "
        self.assertEqual(list(jc.parsers.rsync_s.parse(self.generic_rsync_i.splitlines(), quiet=True)), self.generic_rsync_i_streaming_json)

    def test_rsync_s_ivvv_centos_7_7(self):
        if False:
            return 10
        "\n        Test 'rsync -ivvv' on Centos 7.7\n        "
        self.assertEqual(list(jc.parsers.rsync_s.parse(self.centos_7_7_rsync_ivvv.splitlines(), quiet=True)), self.centos_7_7_rsync_ivvv_streaming_json)

    def test_rsync_s_ivvv_osx_10_14_6(self):
        if False:
            i = 10
            return i + 15
        "\n        Test 'rsync -ivvv' on OSX 10.14.6\n        "
        self.assertEqual(list(jc.parsers.rsync_s.parse(self.osx_10_14_6_rsync_ivvv.splitlines(), quiet=True)), self.osx_10_14_6_rsync_ivvv_streaming_json)

    def test_rsync_s_ivvv_nochange_centos_7_7(self):
        if False:
            print('Hello World!')
        "\n        Test 'rsync -ivvv' on Centos 7.7 with no file changes\n        "
        self.assertEqual(list(jc.parsers.rsync_s.parse(self.centos_7_7_rsync_ivvv_nochange.splitlines(), quiet=True)), self.centos_7_7_rsync_ivvv_nochange_streaming_json)

    def test_rsync_s_ivvv_nochange_osx_10_14_6(self):
        if False:
            return 10
        "\n        Test 'rsync -ivvv' on OSX 10.14.6 with no file changes\n        "
        self.assertEqual(list(jc.parsers.rsync_s.parse(self.osx_10_14_6_rsync_ivvv_nochange.splitlines(), quiet=True)), self.osx_10_14_6_rsync_ivvv_nochange_streaming_json)

    def test_rsync_s_i_logfile_centos_7_7(self):
        if False:
            while True:
                i = 10
        "\n        Test 'rsync -i --logfile=xxx' on Centos 7.7\n        "
        self.assertEqual(list(jc.parsers.rsync_s.parse(self.centos_7_7_rsync_i_logfile.splitlines(), quiet=True)), self.centos_7_7_rsync_i_logfile_streaming_json)

    def test_rsync_s_i_v_logfile_centos_7_7(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Test 'rsync -iv --logfile=xxx' on Centos 7.7\n        "
        self.assertEqual(list(jc.parsers.rsync_s.parse(self.centos_7_7_rsync_i_v_logfile.splitlines(), quiet=True)), self.centos_7_7_rsync_i_v_logfile_streaming_json)

    def test_rsync_s_i_vv_logfile_centos_7_7(self):
        if False:
            return 10
        "\n        Test 'rsync -ivv --logfile=xxx' on Centos 7.7\n        "
        self.assertEqual(list(jc.parsers.rsync_s.parse(self.centos_7_7_rsync_i_vv_logfile.splitlines(), quiet=True)), self.centos_7_7_rsync_i_vv_logfile_streaming_json)

    def test_rsync_s_i_vvv_logfile_centos_7_7(self):
        if False:
            print('Hello World!')
        "\n        Test 'rsync -ivvv --logfile=xxx' on Centos 7.7\n        "
        self.assertEqual(list(jc.parsers.rsync_s.parse(self.centos_7_7_rsync_i_vvv_logfile.splitlines(), quiet=True)), self.centos_7_7_rsync_i_vvv_logfile_streaming_json)

    def test_rsync_s_i_vvv_logfile_nochange_centos_7_7(self):
        if False:
            while True:
                i = 10
        "\n        Test 'rsync -ivvv --logfile=xxx' on Centos 7.7 with no file changes\n        "
        self.assertEqual(list(jc.parsers.rsync_s.parse(self.centos_7_7_rsync_i_vvv_logfile_nochange.splitlines(), quiet=True)), self.centos_7_7_rsync_i_vvv_logfile_nochange_streaming_json)

    def test_rsync_s_i_vvv_logfile_nochange_osx_10_14_6(self):
        if False:
            i = 10
            return i + 15
        "\n        Test 'rsync -ivvv --logfile=xxx' on OSX 10.14.6 with no file changes\n        "
        self.assertEqual(list(jc.parsers.rsync_s.parse(self.osx_10_14_6_rsync_i_vvv_logfile_nochange.splitlines(), quiet=True)), self.osx_10_14_6_rsync_i_vvv_logfile_nochange_streaming_json)
if __name__ == '__main__':
    unittest.main()