from __future__ import absolute_import
from __future__ import print_function
from twisted.trial import unittest
from buildbot_worker.commands import registry
from buildbot_worker.commands import shell

class Registry(unittest.TestCase):

    def test_getFactory(self):
        if False:
            i = 10
            return i + 15
        factory = registry.getFactory('shell')
        self.assertEqual(factory, shell.WorkerShellCommand)

    def test_getFactory_KeyError(self):
        if False:
            return 10
        with self.assertRaises(KeyError):
            registry.getFactory('nosuchcommand')

    def test_getAllCommandNames(self):
        if False:
            i = 10
            return i + 15
        self.assertTrue('shell' in registry.getAllCommandNames())

    def test_all_commands_exist(self):
        if False:
            print('Hello World!')
        for n in registry.getAllCommandNames():
            registry.getFactory(n)