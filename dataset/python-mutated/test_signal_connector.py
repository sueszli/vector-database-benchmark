import concurrent.futures
import os
import signal
from unittest import mock
from unittest.mock import Mock
import pytest
from lightning.fabric.plugins.environments import SLURMEnvironment
from lightning.fabric.utilities.imports import _IS_WINDOWS
from lightning.pytorch import Trainer
from lightning.pytorch.demos.boring_classes import BoringModel
from lightning.pytorch.trainer.connectors.signal_connector import _SignalConnector
from lightning.pytorch.utilities.exceptions import SIGTERMException
from tests_pytorch.helpers.runif import RunIf

@RunIf(skip_windows=True)
def test_signal_handlers_restored_in_teardown():
    if False:
        for i in range(10):
            print('nop')
    'Test that the SignalConnector restores the previously configured handler on teardown.'
    assert signal.getsignal(signal.SIGTERM) is signal.SIG_DFL
    trainer = Trainer(plugins=SLURMEnvironment())
    connector = _SignalConnector(trainer)
    connector.register_signal_handlers()
    assert signal.getsignal(signal.SIGTERM) is not signal.SIG_DFL
    connector.teardown()
    assert signal.getsignal(signal.SIGTERM) is signal.SIG_DFL

@RunIf(skip_windows=True)
def test_sigterm_handler_can_be_added(tmpdir):
    if False:
        while True:
            i = 10
    handler_ran = False

    def handler(*_):
        if False:
            i = 10
            return i + 15
        nonlocal handler_ran
        handler_ran = True
    signal.signal(signal.SIGTERM, handler)

    class TestModel(BoringModel):

        def training_step(self, batch, batch_idx):
            if False:
                return 10
            os.kill(os.getpid(), signal.SIGTERM)
    model = TestModel()
    trainer = Trainer(default_root_dir=tmpdir, max_epochs=1, limit_train_batches=2, limit_val_batches=0)
    assert not trainer.received_sigterm
    assert not handler_ran
    with pytest.raises(SIGTERMException):
        trainer.fit(model)
    assert trainer.received_sigterm
    assert handler_ran
    signal.signal(signal.SIGTERM, signal.SIG_DFL)

@RunIf(skip_windows=True)
@pytest.mark.parametrize('auto_requeue', [True, False])
@pytest.mark.parametrize('requeue_signal', [signal.SIGUSR1, signal.SIGUSR2, signal.SIGHUP] if not _IS_WINDOWS else [])
def test_auto_requeue_custom_signal_flag(auto_requeue, requeue_signal):
    if False:
        for i in range(10):
            print('nop')
    trainer = Trainer(plugins=[SLURMEnvironment(auto_requeue=auto_requeue, requeue_signal=requeue_signal)])
    connector = _SignalConnector(trainer)
    connector.register_signal_handlers()
    if auto_requeue:
        sigterm_handlers = signal.getsignal(signal.SIGTERM).signal_handlers
        assert len(sigterm_handlers) == 2
        assert sigterm_handlers[1].__qualname__ == '_SignalConnector._sigterm_handler_fn'
        sigusr_handlers = signal.getsignal(requeue_signal).signal_handlers
        assert len(sigusr_handlers) == 1
        assert sigusr_handlers[0].__qualname__ == '_SignalConnector._slurm_sigusr_handler_fn'
    else:
        sigterm_handlers = signal.getsignal(signal.SIGTERM).signal_handlers
        assert len(sigterm_handlers) == 1
        assert sigterm_handlers[0].__qualname__ == '_SignalConnector._sigterm_notifier_fn'
        assert signal.getsignal(requeue_signal) is signal.SIG_DFL
    connector.teardown()

@RunIf(skip_windows=True)
@mock.patch('lightning.pytorch.trainer.connectors.signal_connector.call')
@mock.patch('lightning.pytorch.trainer.Trainer.save_checkpoint', mock.MagicMock())
@mock.patch.dict(os.environ, {'SLURM_JOB_ID': '12345'})
def test_auto_requeue_job(call_mock):
    if False:
        i = 10
        return i + 15
    call_mock.return_value = 0
    trainer = Trainer(plugins=[SLURMEnvironment()])
    connector = _SignalConnector(trainer)
    connector._slurm_sigusr_handler_fn(None, None)
    call_mock.assert_called_once_with(['scontrol', 'requeue', '12345'])

@RunIf(skip_windows=True)
@mock.patch('lightning.pytorch.trainer.connectors.signal_connector.call')
@mock.patch('lightning.pytorch.trainer.Trainer.save_checkpoint', mock.MagicMock())
@mock.patch.dict(os.environ, {'SLURM_JOB_ID': '12346', 'SLURM_ARRAY_JOB_ID': '12345', 'SLURM_ARRAY_TASK_ID': '2'})
def test_auto_requeue_array_job(call_mock):
    if False:
        return 10
    call_mock.return_value = 0
    trainer = Trainer(plugins=[SLURMEnvironment()])
    connector = _SignalConnector(trainer)
    connector._slurm_sigusr_handler_fn(None, None)
    call_mock.assert_called_once_with(['scontrol', 'requeue', '12345_2'])

def _registering_signals():
    if False:
        print('Hello World!')
    trainer = Trainer()
    trainer._signal_connector.register_signal_handlers()

@RunIf(skip_windows=True)
def test_signal_connector_in_thread():
    if False:
        for i in range(10):
            print('nop')
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        for future in concurrent.futures.as_completed([executor.submit(_registering_signals)]):
            assert future.exception() is None

def signal_handler():
    if False:
        i = 10
        return i + 15
    pass

class SignalHandlers:

    def signal_handler(self):
        if False:
            for i in range(10):
                print('nop')
        pass

@pytest.mark.parametrize(('handler', 'expected_return'), [(None, False), (signal.Handlers.SIG_IGN, True), (signal.Handlers.SIG_DFL, False), (signal_handler, True), (SignalHandlers().signal_handler, True)])
def test_has_already_handler(handler, expected_return):
    if False:
        for i in range(10):
            print('nop')
    'Test that the SignalConnector detects whether a signal handler is already attached.'
    with mock.patch('lightning.pytorch.trainer.connectors.signal_connector.signal.getsignal', return_value=handler):
        assert _SignalConnector._has_already_handler(signal.SIGTERM) is expected_return

def test_sigterm_notifier_fn():
    if False:
        i = 10
        return i + 15
    trainer = Mock()
    launcher = Mock()
    trainer.strategy.launcher = launcher
    connector = _SignalConnector(trainer)
    assert not connector.received_sigterm
    connector._sigterm_notifier_fn(signal.SIGTERM, Mock())
    launcher.kill.assert_called_once_with(15)
    assert connector.received_sigterm
    launcher.reset_mock()
    connector._sigterm_notifier_fn(signal.SIGTERM, Mock())
    launcher.kill.assert_not_called()