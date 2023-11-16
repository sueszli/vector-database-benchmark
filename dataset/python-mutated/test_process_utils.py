from unittest import TestCase, mock
from mycroft.util.process_utils import _update_log_level, bus_logging_status, create_daemon, ProcessStatus, StatusCallbackMap

class TestCreateDaemon(TestCase):

    def test_create(self):
        if False:
            for i in range(10):
                print('nop')
        'Make sure deamon thread is created, and runs the expected function.\n        '
        thread_ran = False

        def thread_func():
            if False:
                while True:
                    i = 10
            nonlocal thread_ran
            thread_ran = True
        thread = create_daemon(thread_func)
        self.assertTrue(thread.daemon)
        self.assertTrue(thread_ran)
        thread.join()

    def test_create_with_args(self):
        if False:
            print('Hello World!')
        'Check that the args and kwargs is passed to the thread function.'
        test_args = (1, 2, 3)
        test_kwargs = {'meaning': 42, 'borg': '7 of 9'}
        passed_args = None
        passed_kwargs = None

        def thread_func(*args, **kwargs):
            if False:
                i = 10
                return i + 15
            nonlocal passed_args
            nonlocal passed_kwargs
            passed_args = args
            passed_kwargs = kwargs
        thread = create_daemon(thread_func, test_args, test_kwargs)
        thread.join()
        self.assertEqual(test_args, passed_args)
        self.assertEqual(test_kwargs, passed_kwargs)

@mock.patch('mycroft.util.process_utils.LOG')
class TestUpdateLogLevel(TestCase):

    def test_no_data(self, mock_log):
        if False:
            i = 10
            return i + 15
        mock_log.level = 'UNSET'
        log_msg = {'msg_type': 'mycroft.debug.log', 'data': {}}
        _update_log_level(log_msg, 'Test')
        self.assertEqual(mock_log.level, 'UNSET')

    def test_set_debug(self, mock_log):
        if False:
            while True:
                i = 10
        mock_log.level = 'UNSET'
        log_msg = {'type': 'mycroft.debug.log', 'data': {'level': 'DEBUG'}}
        _update_log_level(log_msg, 'Test')
        self.assertEqual(mock_log.level, 'DEBUG')

    def test_set_lowecase_debug(self, mock_log):
        if False:
            for i in range(10):
                print('nop')
        mock_log.level = 'UNSET'
        log_msg = {'type': 'mycroft.debug.log', 'data': {'level': 'debug'}}
        _update_log_level(log_msg, 'Test')
        self.assertEqual(mock_log.level, 'DEBUG')

    def test_set_invalid_level(self, mock_log):
        if False:
            while True:
                i = 10
        mock_log.level = 'UNSET'
        log_msg = {'type': 'mycroft.debug.log', 'data': {'level': 'snowcrash'}}
        _update_log_level(log_msg, 'Test')
        self.assertEqual(mock_log.level, 'UNSET')

    def test_set_bus_logging(self, mock_log):
        if False:
            while True:
                i = 10
        mock_log.level = 'UNSET'
        log_msg = {'type': 'mycroft.debug.log', 'data': {'bus': True}}
        self.assertFalse(bus_logging_status())
        _update_log_level(log_msg, 'Test')
        self.assertTrue(bus_logging_status())

def create_mock_message(msg_type):
    if False:
        while True:
            i = 10
    'Creates a mock with members matching a messagebus Message.'
    m = mock.Mock()
    m.msg_type = msg_type
    m.data = {}
    m.context = {}
    return m

class TestProcessStatus(TestCase):

    def test_callbacks(self):
        if False:
            while True:
                i = 10
        'Assert that callbacks are called as expected.'
        started = False
        alive = False
        ready = False
        stopping = False
        error = False

        def started_hook():
            if False:
                for i in range(10):
                    print('nop')
            nonlocal started
            started = True

        def alive_hook():
            if False:
                for i in range(10):
                    print('nop')
            nonlocal alive
            alive = True

        def ready_hook():
            if False:
                for i in range(10):
                    print('nop')
            nonlocal ready
            ready = True

        def stopping_hook():
            if False:
                i = 10
                return i + 15
            nonlocal stopping
            stopping = True

        def error_hook(err):
            if False:
                for i in range(10):
                    print('nop')
            nonlocal error
            error = err
        callbacks = StatusCallbackMap(on_started=started_hook, on_alive=alive_hook, on_ready=ready_hook, on_stopping=stopping_hook, on_error=error_hook)
        status = ProcessStatus('test', mock.Mock(), callbacks)
        status.set_started()
        self.assertTrue(started)
        status.set_alive()
        self.assertTrue(alive)
        status.set_ready()
        self.assertTrue(ready)
        status.set_stopping()
        self.assertTrue(stopping)
        err_msg = 'Test error'
        status.set_error(err_msg)
        self.assertEqual(err_msg, error)

    def test_init_status(self):
        if False:
            return 10
        'Check that the status is neither alive nor ready after init.'
        status = ProcessStatus('test', mock.Mock())
        self.assertFalse(status.check_alive())
        self.assertFalse(status.check_ready())

    def test_alive_status(self):
        if False:
            while True:
                i = 10
        status = ProcessStatus('test', mock.Mock())
        status.set_alive()
        self.assertTrue(status.check_alive())
        self.assertFalse(status.check_ready())

    def test_ready_status(self):
        if False:
            while True:
                i = 10
        'Check that alive and ready reports correctly.'
        status = ProcessStatus('test', mock.Mock())
        status.set_alive()
        status.set_ready()
        self.assertTrue(status.check_alive())
        self.assertTrue(status.check_ready())

    def test_direct_to_ready_status(self):
        if False:
            return 10
        'Ensure that process status indicates alive if only ready is set.'
        status = ProcessStatus('test', mock.Mock())
        status.set_ready()
        self.assertTrue(status.check_alive())
        self.assertTrue(status.check_ready())

    def test_error_status(self):
        if False:
            i = 10
            return i + 15
        'Ensure that error resets the status and to not alive or ready.'
        status = ProcessStatus('test', mock.Mock())
        status.set_ready()
        status.set_error()
        self.assertFalse(status.check_alive())
        self.assertFalse(status.check_ready())

    def test_ready_message(self):
        if False:
            i = 10
            return i + 15
        'Assert that ready message contains the correct status.'
        status = ProcessStatus('test', mock.Mock())
        msg = create_mock_message('mycroft.test.all_loaded')
        status.check_ready(msg)
        msg.response.assert_called_with(data={'status': False})
        status.set_ready()
        msg = create_mock_message('mycroft.test.all_loaded')
        status.check_ready(msg)
        msg.response.assert_called_with(data={'status': True})

    def test_is_alive__message(self):
        if False:
            print('Hello World!')
        'Assert that is_alive message contains the correct status.'
        status = ProcessStatus('test', mock.Mock())
        status.set_started()
        msg = create_mock_message('mycroft.test.is_alive')
        status.check_alive(msg)
        msg.response.assert_called_with(data={'status': False})
        status.set_ready()
        msg = create_mock_message('mycroft.test.is_isalive')
        status.check_alive(msg)
        msg.response.assert_called_with(data={'status': True})