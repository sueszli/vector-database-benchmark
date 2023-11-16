import pytest
from rocketry.args.builtin import SimpleArg
from rocketry.conditions.task.task import TaskStarted
from rocketry.core.parameters.parameters import Parameters
from rocketry.tasks import FuncTask
from rocketry.conditions import SchedulerCycles, AlwaysFalse

def run_succeeding():
    if False:
        print('Hello World!')
    pass

def run_parametrized(arg=SimpleArg('incorrect')):
    if False:
        for i in range(10):
            print('nop')
    assert arg == 'correct'

@pytest.mark.parametrize('execution', ['main', 'thread', 'process'])
def test_set_running(execution, session):
    if False:
        return 10
    task = FuncTask(run_succeeding, start_cond=AlwaysFalse(), name='task', execution=execution, session=session)
    assert task.batches == []
    task.run()
    assert task.batches == [Parameters()]
    session.config.shut_cond = SchedulerCycles() >= 5
    session.start()
    logger = task.logger
    assert 1 == logger.filter_by(action='run').count()
    assert 1 == logger.filter_by(action='success').count()
    assert 0 == logger.filter_by(action='fail').count()
    assert len(task.batches) == 0

@pytest.mark.parametrize('execution', ['main', 'thread', 'process'])
def test_set_running_with_params(execution, session):
    if False:
        while True:
            i = 10
    task = FuncTask(run_parametrized, start_cond=AlwaysFalse(), name='task', execution=execution, session=session)
    task.run(arg='correct')
    task.run(arg='correct')
    task.run(arg='incorrect')
    assert task.batches == [Parameters({'arg': 'correct'}), Parameters({'arg': 'correct'}), Parameters({'arg': 'incorrect'})]
    session.config.shut_cond = TaskStarted(task=task) == 3
    session.start()
    logger = task.logger
    assert 3 == logger.filter_by(action='run').count()
    assert 2 == logger.filter_by(action='success').count()
    assert 1 == logger.filter_by(action='fail').count()
    assert len(task.batches) == 0

@pytest.mark.parametrize('execution', ['main', 'thread', 'process'])
def test_set_running_disabled(execution, session):
    if False:
        i = 10
        return i + 15
    task = FuncTask(run_succeeding, start_cond=AlwaysFalse(), name='task', execution=execution, session=session)
    task.disabled = True
    task.run()
    session.config.shut_cond = SchedulerCycles() >= 5
    session.start()
    assert task.batches == []
    logger = task.logger
    assert 1 == logger.filter_by(action='run').count()
    assert 1 == logger.filter_by(action='success').count()
    assert task.disabled

@pytest.mark.parametrize('execution', ['main', 'thread', 'process'])
def test_task_force_run(execution, session):
    if False:
        print('Hello World!')
    task = FuncTask(run_succeeding, start_cond=AlwaysFalse(), name='task', execution=execution, session=session)
    with pytest.warns(DeprecationWarning):
        task.force_run = True
    session.config.shut_cond = SchedulerCycles() >= 5
    session.start()
    logger = task.logger
    assert 1 == logger.filter_by(action='run').count()
    assert not task.force_run

@pytest.mark.parametrize('execution', ['main', 'thread', 'process'])
def test_task_force_disabled(execution, session):
    if False:
        print('Hello World!')
    task = FuncTask(run_succeeding, start_cond=AlwaysFalse(), name='task', execution=execution, session=session)
    task.disabled = True
    with pytest.warns(DeprecationWarning):
        task.force_run = True
    session.config.shut_cond = SchedulerCycles() >= 5
    session.start()
    assert task.batches == []
    logger = task.logger
    assert 1 == logger.filter_by(action='run').count()
    assert 1 == logger.filter_by(action='success').count()
    assert task.disabled
    assert not task.force_run