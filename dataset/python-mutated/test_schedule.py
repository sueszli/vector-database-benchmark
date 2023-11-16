import numpy as np
from neon.optimizers import Schedule, ExpSchedule, PowerSchedule, StepSchedule, ShiftSchedule
from utils import allclose_with_out

def test_schedule(backend_default):
    if False:
        print('Hello World!')
    '\n    Test constant rate, fixed step and various modes of programmable steps.\n    '
    lr_init = 0.1
    sch = Schedule()
    for epoch in range(10):
        lr = sch.get_learning_rate(learning_rate=lr_init, epoch=epoch)
        assert lr == lr_init
    step_config = 2
    change = 0.5
    sch = Schedule(step_config=step_config, change=change)
    for epoch in range(10):
        lr = sch.get_learning_rate(learning_rate=lr_init, epoch=epoch)
        lr2 = sch.get_learning_rate(learning_rate=lr_init, epoch=epoch)
        assert allclose_with_out(lr, lr_init * change ** np.floor(epoch // step_config))
        assert allclose_with_out(lr2, lr_init * change ** np.floor(epoch // step_config))
    sch = Schedule(step_config=[2, 3], change=0.1)
    assert allclose_with_out(0.1, sch.get_learning_rate(learning_rate=0.1, epoch=0))
    assert allclose_with_out(0.1, sch.get_learning_rate(learning_rate=0.1, epoch=1))
    assert allclose_with_out(0.01, sch.get_learning_rate(learning_rate=0.1, epoch=2))
    assert allclose_with_out(0.01, sch.get_learning_rate(learning_rate=0.1, epoch=2))
    assert allclose_with_out(0.001, sch.get_learning_rate(learning_rate=0.1, epoch=3))
    assert allclose_with_out(0.001, sch.get_learning_rate(learning_rate=0.1, epoch=4))

def test_step_schedule(backend_default):
    if False:
        for i in range(10):
            print('nop')
    '\n    Test the StepSchedule class\n    '
    step_config = [1, 4, 5]
    change = [0.1, 0.3, 0.4]
    sch = StepSchedule(step_config=step_config, change=change)
    target_lr = [1.0, 0.1, 0.1, 0.1, 0.3, 0.4, 0.4, 0.4, 0.4]
    for (e, lr) in enumerate(target_lr):
        assert allclose_with_out(lr, sch.get_learning_rate(learning_rate=1.0, epoch=e))

def test_power_schedule(backend_default):
    if False:
        for i in range(10):
            print('nop')
    '\n    Test the PowerSchedule class\n    '
    sch = PowerSchedule(step_config=2, change=0.5)
    target_lr = [1.0, 1.0, 0.5, 0.5, 0.25, 0.25, 0.125, 0.125]
    for (e, lr) in enumerate(target_lr):
        assert allclose_with_out(lr, sch.get_learning_rate(learning_rate=1.0, epoch=e))

def test_exp_schedule(backend_default):
    if False:
        print('Hello World!')
    '\n    Test exponential learning rate schedule\n    '
    lr_init = 0.1
    decay = 0.01
    sch = ExpSchedule(decay)
    for epoch in range(10):
        lr = sch.get_learning_rate(learning_rate=lr_init, epoch=epoch)
        assert allclose_with_out(lr, lr_init / (1.0 + decay * epoch))

def test_shift_schedule(backend_default):
    if False:
        return 10
    '\n    Test binary shift learning rate schedule\n    '
    lr_init = 0.1
    interval = 1
    sch = ShiftSchedule(interval)
    for epoch in range(10):
        lr = sch.get_learning_rate(learning_rate=lr_init, epoch=epoch)
        assert allclose_with_out(lr, lr_init / 2 ** epoch)