import json
import logging
import os
import shutil
import struct
from tempfile import mkdtemp, mkstemp
import unittest
from unittest.mock import Mock
from PyQt6 import QtCore
from picard import config, log
from picard.i18n import setup_gettext
from picard.releasegroup import ReleaseGroup

class FakeThreadPool(QtCore.QObject):

    def start(self, runnable, priority):
        if False:
            return 10
        runnable.run()

class FakeTagger(QtCore.QObject):
    tagger_stats_changed = QtCore.pyqtSignal()

    def __init__(self):
        if False:
            return 10
        QtCore.QObject.__init__(self)
        QtCore.QObject.config = config
        QtCore.QObject.log = log
        self.tagger_stats_changed.connect(self.emit)
        self.exit_cleanup = []
        self.files = {}
        self.stopping = False
        self.thread_pool = FakeThreadPool()
        self.priority_thread_pool = FakeThreadPool()

    def register_cleanup(self, func):
        if False:
            print('Hello World!')
        self.exit_cleanup.append(func)

    def run_cleanup(self):
        if False:
            for i in range(10):
                print('nop')
        for f in self.exit_cleanup:
            f()

    def emit(self, *args):
        if False:
            return 10
        pass

    def get_release_group_by_id(self, rg_id):
        if False:
            for i in range(10):
                print('nop')
        return ReleaseGroup(rg_id)

class PicardTestCase(unittest.TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        log.set_level(logging.DEBUG)
        setup_gettext(None, 'C')
        self.tagger = FakeTagger()
        QtCore.QObject.tagger = self.tagger
        QtCore.QCoreApplication.instance = lambda : self.tagger
        self.addCleanup(self.tagger.run_cleanup)
        self.init_config()

    @staticmethod
    def init_config():
        if False:
            for i in range(10):
                print('nop')
        fake_config = Mock()
        fake_config.setting = {}
        fake_config.persist = {}
        fake_config.profiles = {}
        config.config = fake_config
        config.setting = fake_config.setting
        config.persist = fake_config.persist
        config.profiles = fake_config.profiles

    @staticmethod
    def set_config_values(setting=None, persist=None, profiles=None):
        if False:
            for i in range(10):
                print('nop')
        if setting:
            for (key, value) in setting.items():
                config.config.setting[key] = value
        if persist:
            for (key, value) in persist.items():
                config.config.persist[key] = value
        if profiles:
            for (key, value) in profiles.items():
                config.config.profiles[key] = value

    def mktmpdir(self, ignore_errors=False):
        if False:
            return 10
        tmpdir = mkdtemp(suffix=self.__class__.__name__)
        self.addCleanup(shutil.rmtree, tmpdir, ignore_errors=ignore_errors)
        return tmpdir

    def copy_file_tmp(self, filepath, ext):
        if False:
            while True:
                i = 10
        (fd, copy) = mkstemp(suffix=ext)
        os.close(fd)
        self.addCleanup(self.remove_file_tmp, copy)
        shutil.copy(filepath, copy)
        return copy

    @staticmethod
    def remove_file_tmp(filepath):
        if False:
            while True:
                i = 10
        if os.path.isfile(filepath):
            os.unlink(filepath)

def get_test_data_path(*paths):
    if False:
        print('Hello World!')
    return os.path.join('test', 'data', *paths)

def create_fake_png(extra):
    if False:
        return 10
    "Creates fake PNG data that satisfies Picard's internal image type detection"
    return b'\x89PNG\r\n\x1a\n' + b'a' * 4 + b'IHDR' + struct.pack('>LL', 100, 100) + extra

def load_test_json(filename):
    if False:
        for i in range(10):
            print('nop')
    with open(get_test_data_path('ws_data', filename), encoding='utf-8') as f:
        return json.load(f)