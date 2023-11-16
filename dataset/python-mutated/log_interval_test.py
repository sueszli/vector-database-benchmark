import unittest
from mock import MagicMock
from mock import patch
from pokemongo_bot.cell_workers import MoveToFort
from tests import FakeBot

class LogDelayTestCase(unittest.TestCase):
    config = {'enabled': 'true', 'lure_attraction': 'true', 'lure_max_distance': 2000, 'walker': 'StepWalker', 'log_interval': 3}

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.bot = FakeBot()
        self.bot.event_manager = MagicMock()
        self.worker = MoveToFort(self.bot, self.config)

    def test_read_correct_delay_config(self):
        if False:
            for i in range(10):
                print('nop')
        self.worker.config['log_interval'] = 3
        self.assertEqual(self.config.get('log_interval'), 3)

    def test_log_with_no_delay(self):
        if False:
            print('Hello World!')
        self.worker.config['log_interval'] = 3
        self.worker.emit_event('moving_to_fort', formatted='just an example')
        self.worker.emit_event('moving_to_fort', formatted='just an example')
        self.worker.last_log_time -= 2
        self.worker.emit_event('moving_to_fort', formatted='just an example')
        self.worker.emit_event('moving_to_fort', formatted='just an example')
        self.assertEqual(self.bot.event_manager.emit.call_count, 0)
        assert not self.bot.event_manager.emit.called

    def test_correct_delay_wait(self):
        if False:
            while True:
                i = 10
        self.worker.config['log_interval'] = 2
        self.worker.last_log_time -= self.worker.config['log_interval']
        for number_of_checks in range(10):
            self.worker.emit_event('moving_to_fort', formatted='just an example')
            self.worker.last_log_time -= 2
        self.assertEqual(self.bot.event_manager.emit.call_count, 10)