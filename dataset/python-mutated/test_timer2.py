import sys
import time
from unittest.mock import Mock, call, patch
from celery.utils import timer2 as timer2

class test_Timer:

    def test_enter_after(self):
        if False:
            return 10
        t = timer2.Timer()
        try:
            done = [False]

            def set_done():
                if False:
                    print('Hello World!')
                done[0] = True
            t.call_after(0.3, set_done)
            mss = 0
            while not done[0]:
                if mss >= 2.0:
                    raise Exception('test timed out')
                time.sleep(0.1)
                mss += 0.1
        finally:
            t.stop()

    def test_exit_after(self):
        if False:
            i = 10
            return i + 15
        t = timer2.Timer()
        t.call_after = Mock()
        t.exit_after(0.3, priority=10)
        t.call_after.assert_called_with(0.3, sys.exit, 10)

    def test_ensure_started_not_started(self):
        if False:
            i = 10
            return i + 15
        t = timer2.Timer()
        t.running = True
        t.start = Mock()
        t.ensure_started()
        t.start.assert_not_called()
        t.running = False
        t.on_start = Mock()
        t.ensure_started()
        t.on_start.assert_called_with(t)
        t.start.assert_called_with()

    @patch('celery.utils.timer2.sleep')
    @patch('os._exit')
    def test_on_tick(self, _exit, sleep):
        if False:
            while True:
                i = 10

        def next_entry_side_effect():
            if False:
                i = 10
                return i + 15
            for _ in range(3):
                yield 3.33
            while True:
                yield getattr(t, '_Timer__is_shutdown').set()
        on_tick = Mock(name='on_tick')
        t = timer2.Timer(on_tick=on_tick)
        t._next_entry = Mock(name='_next_entry', side_effect=next_entry_side_effect())
        t.run()
        sleep.assert_called_with(3.33)
        on_tick.assert_has_calls([call(3.33), call(3.33), call(3.33)])
        _exit.assert_not_called()

    @patch('os._exit')
    def test_thread_crash(self, _exit):
        if False:
            while True:
                i = 10
        t = timer2.Timer()
        t._next_entry = Mock()
        t._next_entry.side_effect = OSError(131)
        t.run()
        _exit.assert_called_with(1)

    def test_gc_race_lost(self):
        if False:
            for i in range(10):
                print('nop')
        t = timer2.Timer()
        with patch.object(t, '_Timer__is_stopped') as mock_stop_event:
            with patch.object(t, 'running', new=False):
                t.stop()
            mock_stop_event.set.side_effect = TypeError()
            t.run()
        mock_stop_event.set.assert_called_with()

    def test_test_enter(self):
        if False:
            return 10
        t = timer2.Timer()
        t._do_enter = Mock()
        e = Mock()
        t.enter(e, 13, 0)
        t._do_enter.assert_called_with('enter_at', e, 13, priority=0)

    def test_test_enter_after(self):
        if False:
            print('Hello World!')
        t = timer2.Timer()
        t._do_enter = Mock()
        t.enter_after()
        t._do_enter.assert_called_with('enter_after')

    def test_cancel(self):
        if False:
            print('Hello World!')
        t = timer2.Timer()
        tref = Mock()
        t.cancel(tref)
        tref.cancel.assert_called_with()