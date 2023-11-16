import os
import unittest
import json
from typing import Dict
import jc.parsers.proc_pid_fdinfo
THIS_DIR = os.path.dirname(os.path.abspath(__file__))

class MyTests(unittest.TestCase):
    f_in: Dict = {}
    f_json: Dict = {}

    @classmethod
    def setUpClass(cls):
        if False:
            return 10
        fixtures = {'proc_pid_fdinfo': ('fixtures/linux-proc/pid_fdinfo', 'fixtures/linux-proc/pid_fdinfo.json'), 'proc_pid_fdinfo_dma': ('fixtures/linux-proc/pid_fdinfo_dma', 'fixtures/linux-proc/pid_fdinfo_dma.json'), 'proc_pid_fdinfo_epoll': ('fixtures/linux-proc/pid_fdinfo_epoll', 'fixtures/linux-proc/pid_fdinfo_epoll.json'), 'proc_pid_fdinfo_fanotify': ('fixtures/linux-proc/pid_fdinfo_fanotify', 'fixtures/linux-proc/pid_fdinfo_fanotify.json'), 'proc_pid_fdinfo_inotify': ('fixtures/linux-proc/pid_fdinfo_inotify', 'fixtures/linux-proc/pid_fdinfo_inotify.json'), 'proc_pid_fdinfo_timerfd': ('fixtures/linux-proc/pid_fdinfo_timerfd', 'fixtures/linux-proc/pid_fdinfo_timerfd.json')}
        for (file, filepaths) in fixtures.items():
            with open(os.path.join(THIS_DIR, filepaths[0]), 'r', encoding='utf-8') as a, open(os.path.join(THIS_DIR, filepaths[1]), 'r', encoding='utf-8') as b:
                cls.f_in[file] = a.read()
                cls.f_json[file] = json.loads(b.read())

    def test_proc_pid_fdinfo_nodata(self):
        if False:
            i = 10
            return i + 15
        "\n        Test 'proc_pid_fdinfo' with no data\n        "
        self.assertEqual(jc.parsers.proc_pid_fdinfo.parse('', quiet=True), {})

    def test_proc_pid_fdinfo(self):
        if False:
            print('Hello World!')
        "\n        Test '/proc/<pid>/fdinfo/<fd>'\n        "
        self.assertEqual(jc.parsers.proc_pid_fdinfo.parse(self.f_in['proc_pid_fdinfo'], quiet=True), self.f_json['proc_pid_fdinfo'])

    def test_proc_pid_fdinfo_dma(self):
        if False:
            print('Hello World!')
        "\n        Test '/proc/<pid>/fdinfo/<fd>' dma file\n        "
        self.assertEqual(jc.parsers.proc_pid_fdinfo.parse(self.f_in['proc_pid_fdinfo_dma'], quiet=True), self.f_json['proc_pid_fdinfo_dma'])

    def test_proc_pid_fdinfo_epoll(self):
        if False:
            print('Hello World!')
        "\n        Test '/proc/<pid>/fdinfo/<fd>' epoll file\n        "
        self.assertEqual(jc.parsers.proc_pid_fdinfo.parse(self.f_in['proc_pid_fdinfo_epoll'], quiet=True), self.f_json['proc_pid_fdinfo_epoll'])

    def test_proc_pid_fdinfo_fanotify(self):
        if False:
            return 10
        "\n        Test '/proc/<pid>/fdinfo/<fd>' fanotify file\n        "
        self.assertEqual(jc.parsers.proc_pid_fdinfo.parse(self.f_in['proc_pid_fdinfo_fanotify'], quiet=True), self.f_json['proc_pid_fdinfo_fanotify'])

    def test_proc_pid_fdinfo_inotify(self):
        if False:
            print('Hello World!')
        "\n        Test '/proc/<pid>/fdinfo/<fd>' inotify file\n        "
        self.assertEqual(jc.parsers.proc_pid_fdinfo.parse(self.f_in['proc_pid_fdinfo_inotify'], quiet=True), self.f_json['proc_pid_fdinfo_inotify'])

    def test_proc_pid_fdinfo_timerfd(self):
        if False:
            return 10
        "\n        Test '/proc/<pid>/fdinfo/<fd>' timerfd file\n        "
        self.assertEqual(jc.parsers.proc_pid_fdinfo.parse(self.f_in['proc_pid_fdinfo_timerfd'], quiet=True), self.f_json['proc_pid_fdinfo_timerfd'])
if __name__ == '__main__':
    unittest.main()