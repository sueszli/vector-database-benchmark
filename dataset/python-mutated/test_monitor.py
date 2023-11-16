from torch.testing._internal.common_utils import TestCase, run_tests
from datetime import timedelta, datetime
import tempfile
import time
from torch.monitor import Aggregation, Event, log_event, register_event_handler, unregister_event_handler, Stat, TensorboardEventHandler

class TestMonitor(TestCase):

    def test_interval_stat(self) -> None:
        if False:
            while True:
                i = 10
        events = []

        def handler(event):
            if False:
                for i in range(10):
                    print('nop')
            events.append(event)
        handle = register_event_handler(handler)
        s = Stat('asdf', (Aggregation.SUM, Aggregation.COUNT), timedelta(milliseconds=1))
        self.assertEqual(s.name, 'asdf')
        s.add(2)
        for _ in range(100):
            time.sleep(1 / 1000)
            s.add(3)
            if len(events) >= 1:
                break
        self.assertGreaterEqual(len(events), 1)
        unregister_event_handler(handle)

    def test_fixed_count_stat(self) -> None:
        if False:
            print('Hello World!')
        s = Stat('asdf', (Aggregation.SUM, Aggregation.COUNT), timedelta(hours=100), 3)
        s.add(1)
        s.add(2)
        name = s.name
        self.assertEqual(name, 'asdf')
        self.assertEqual(s.count, 2)
        s.add(3)
        self.assertEqual(s.count, 0)
        self.assertEqual(s.get(), {Aggregation.SUM: 6.0, Aggregation.COUNT: 3})

    def test_log_event(self) -> None:
        if False:
            print('Hello World!')
        e = Event(name='torch.monitor.TestEvent', timestamp=datetime.now(), data={'str': 'a string', 'float': 1234.0, 'int': 1234})
        self.assertEqual(e.name, 'torch.monitor.TestEvent')
        self.assertIsNotNone(e.timestamp)
        self.assertIsNotNone(e.data)
        log_event(e)

    def test_event_handler(self) -> None:
        if False:
            while True:
                i = 10
        events = []

        def handler(event: Event) -> None:
            if False:
                for i in range(10):
                    print('nop')
            events.append(event)
        handle = register_event_handler(handler)
        e = Event(name='torch.monitor.TestEvent', timestamp=datetime.now(), data={})
        log_event(e)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0], e)
        log_event(e)
        self.assertEqual(len(events), 2)
        unregister_event_handler(handle)
        log_event(e)
        self.assertEqual(len(events), 2)

class TestMonitorTensorboard(TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        global SummaryWriter, event_multiplexer
        try:
            from torch.utils.tensorboard import SummaryWriter
            from tensorboard.backend.event_processing import plugin_event_multiplexer as event_multiplexer
        except ImportError:
            return self.skipTest('Skip the test since TensorBoard is not installed')
        self.temp_dirs = []

    def create_summary_writer(self):
        if False:
            return 10
        temp_dir = tempfile.TemporaryDirectory()
        self.temp_dirs.append(temp_dir)
        return SummaryWriter(temp_dir.name)

    def tearDown(self):
        if False:
            while True:
                i = 10
        for temp_dir in self.temp_dirs:
            temp_dir.cleanup()

    def test_event_handler(self):
        if False:
            return 10
        with self.create_summary_writer() as w:
            handle = register_event_handler(TensorboardEventHandler(w))
            s = Stat('asdf', (Aggregation.SUM, Aggregation.COUNT), timedelta(hours=1), 5)
            for i in range(10):
                s.add(i)
            self.assertEqual(s.count, 0)
            unregister_event_handler(handle)
        mul = event_multiplexer.EventMultiplexer()
        mul.AddRunsFromDirectory(self.temp_dirs[-1].name)
        mul.Reload()
        scalar_dict = mul.PluginRunToTagToContent('scalars')
        raw_result = {tag: mul.Tensors(run, tag) for (run, run_dict) in scalar_dict.items() for tag in run_dict}
        scalars = {tag: [e.tensor_proto.float_val[0] for e in events] for (tag, events) in raw_result.items()}
        self.assertEqual(scalars, {'asdf.sum': [10], 'asdf.count': [5]})
if __name__ == '__main__':
    run_tests()