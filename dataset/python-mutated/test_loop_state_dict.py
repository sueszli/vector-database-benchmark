from lightning.pytorch.loops import _FitLoop
from lightning.pytorch.trainer.trainer import Trainer

def test_loops_state_dict():
    if False:
        return 10
    trainer = Trainer()
    fit_loop = _FitLoop(trainer)
    state_dict = fit_loop.state_dict()
    new_fit_loop = _FitLoop(trainer)
    new_fit_loop.load_state_dict(state_dict)
    assert fit_loop.state_dict() == new_fit_loop.state_dict()

def test_loops_state_dict_structure():
    if False:
        print('Hello World!')
    trainer = Trainer()
    state_dict = trainer._checkpoint_connector._get_loops_state_dict()
    expected = {'fit_loop': {'state_dict': {}, 'epoch_loop.state_dict': {'_batches_that_stepped': 0}, 'epoch_loop.batch_progress': {'total': {'ready': 0, 'started': 0, 'processed': 0, 'completed': 0}, 'current': {'ready': 0, 'started': 0, 'processed': 0, 'completed': 0}, 'is_last_batch': False}, 'epoch_loop.scheduler_progress': {'total': {'ready': 0, 'completed': 0}, 'current': {'ready': 0, 'completed': 0}}, 'epoch_loop.manual_optimization.state_dict': {}, 'epoch_loop.manual_optimization.optim_step_progress': {'total': {'ready': 0, 'completed': 0}, 'current': {'ready': 0, 'completed': 0}}, 'epoch_loop.automatic_optimization.state_dict': {}, 'epoch_loop.automatic_optimization.optim_progress': {'optimizer': {'step': {'total': {'ready': 0, 'completed': 0}, 'current': {'ready': 0, 'completed': 0}}, 'zero_grad': {'total': {'ready': 0, 'started': 0, 'completed': 0}, 'current': {'ready': 0, 'started': 0, 'completed': 0}}}}, 'epoch_loop.val_loop.state_dict': {}, 'epoch_loop.val_loop.batch_progress': {'total': {'ready': 0, 'started': 0, 'processed': 0, 'completed': 0}, 'current': {'ready': 0, 'started': 0, 'processed': 0, 'completed': 0}, 'is_last_batch': False}, 'epoch_progress': {'total': {'ready': 0, 'started': 0, 'processed': 0, 'completed': 0}, 'current': {'ready': 0, 'started': 0, 'processed': 0, 'completed': 0}}}, 'validate_loop': {'state_dict': {}, 'batch_progress': {'total': {'ready': 0, 'started': 0, 'processed': 0, 'completed': 0}, 'current': {'ready': 0, 'started': 0, 'processed': 0, 'completed': 0}, 'is_last_batch': False}}, 'test_loop': {'state_dict': {}, 'batch_progress': {'total': {'ready': 0, 'started': 0, 'processed': 0, 'completed': 0}, 'current': {'ready': 0, 'started': 0, 'processed': 0, 'completed': 0}, 'is_last_batch': False}}, 'predict_loop': {'state_dict': {}, 'batch_progress': {'total': {'ready': 0, 'started': 0, 'processed': 0, 'completed': 0}, 'current': {'ready': 0, 'started': 0, 'processed': 0, 'completed': 0}}}}
    assert state_dict == expected