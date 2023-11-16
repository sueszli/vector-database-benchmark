import unittest
import json
from pokemongo_bot.base_task import BaseTask

class FakeTask(BaseTask):

    def initialize(self):
        if False:
            while True:
                i = 10
        self.foo = 'foo'

    def work(self):
        if False:
            return 10
        pass

class FakeTaskWithoutInitialize(BaseTask):

    def work(self):
        if False:
            while True:
                i = 10
        pass

class FakeTaskWithoutWork(BaseTask):
    pass

class BaseTaskTest(unittest.TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.bot = {}
        self.config = {}

    def test_initialize_called(self):
        if False:
            while True:
                i = 10
        task = FakeTask(self.bot, self.config)
        self.assertIs(task.bot, self.bot)
        self.assertIs(task.config, self.config)
        self.assertEquals(task.foo, 'foo')

    def test_does_not_throw_without_initialize(self):
        if False:
            return 10
        FakeTaskWithoutInitialize(self.bot, self.config)

    def test_throws_without_work(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertRaisesRegexp(NotImplementedError, 'Missing "work" method', FakeTaskWithoutWork, self.bot, self.config)