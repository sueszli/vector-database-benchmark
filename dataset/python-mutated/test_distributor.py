from unittest.mock import Mock, patch
from synapse.util.distributor import Distributor
from . import unittest

class DistributorTestCase(unittest.TestCase):

    def setUp(self) -> None:
        if False:
            while True:
                i = 10
        self.dist = Distributor()

    def test_signal_dispatch(self) -> None:
        if False:
            print('Hello World!')
        self.dist.declare('alert')
        observer = Mock()
        self.dist.observe('alert', observer)
        self.dist.fire('alert', 1, 2, 3)
        observer.assert_called_with(1, 2, 3)

    def test_signal_catch(self) -> None:
        if False:
            while True:
                i = 10
        self.dist.declare('alarm')
        observers = [Mock() for i in (1, 2)]
        for o in observers:
            self.dist.observe('alarm', o)
        observers[0].side_effect = Exception('Awoogah!')
        with patch('synapse.util.distributor.logger', spec=['warning']) as mock_logger:
            self.dist.fire('alarm', 'Go')
            observers[0].assert_called_once_with('Go')
            observers[1].assert_called_once_with('Go')
            self.assertEqual(mock_logger.warning.call_count, 1)
            self.assertIsInstance(mock_logger.warning.call_args[0][0], str)

    def test_signal_prereg(self) -> None:
        if False:
            print('Hello World!')
        observer = Mock()
        self.dist.observe('flare', observer)
        self.dist.declare('flare')
        self.dist.fire('flare', 4, 5)
        observer.assert_called_with(4, 5)

    def test_signal_undeclared(self) -> None:
        if False:
            while True:
                i = 10

        def code() -> None:
            if False:
                print('Hello World!')
            self.dist.fire('notification')
        self.assertRaises(KeyError, code)