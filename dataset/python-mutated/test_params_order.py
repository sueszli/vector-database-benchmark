from pathlib import Path
from textwrap import dedent
import pytest
from rocketry.args.builtin import SimpleArg
from rocketry.core.condition.base import AlwaysTrue
from rocketry.tasks import FuncTask
from rocketry.conditions import SchedulerCycles

def run_parametrized(arg):
    if False:
        while True:
            i = 10
    assert arg == 'correct'

def run_parametrized_correct(arg=SimpleArg('correct')):
    if False:
        print('Hello World!')
    assert arg == 'correct'

def run_parametrized_incorrect(arg=SimpleArg('incorrect')):
    if False:
        for i in range(10):
            print('nop')
    assert arg == 'correct'

@pytest.mark.parametrize('execution', ['main', 'thread', 'process'])
def test_batch_favored(execution, session):
    if False:
        return 10
    session.parameters['arg'] = 'incorrect'
    task = FuncTask(run_parametrized_incorrect, start_cond=AlwaysTrue(), name='task', execution=execution, session=session, parameters={'arg': 'incorrect'})
    task.run(arg='correct')
    session.config.shut_cond = SchedulerCycles() == 1
    session.start()
    logger = task.logger
    assert 1 == logger.filter_by(action='run').count()
    assert 1 == logger.filter_by(action='success').count()

@pytest.mark.parametrize('delayed', [True, False])
@pytest.mark.parametrize('execution', ['main', 'thread', 'process'])
def test_task_favored(execution, session, tmpdir, delayed):
    if False:
        print('Hello World!')
    session.parameters['arg'] = 'incorrect'
    if delayed:
        funcfile = tmpdir.join('script_task_favored.py')
        funcfile.write(dedent('\n            from rocketry.args import SimpleArg\n            def run_parametrized_incorrect(arg=SimpleArg("incorrect")):\n                assert arg == "correct"\n        '))
        task = FuncTask(path=Path(funcfile), func_name='run_parametrized_incorrect', start_cond=AlwaysTrue(), name='task', execution=execution, session=session, parameters={'arg': 'correct'})
    else:
        task = FuncTask(run_parametrized_incorrect, start_cond=AlwaysTrue(), name='task', execution=execution, session=session, parameters={'arg': 'correct'})
    session.config.shut_cond = SchedulerCycles() == 1
    session.start()
    logger = task.logger
    assert 1 == logger.filter_by(action='run').count()
    assert 1 == logger.filter_by(action='success').count()

@pytest.mark.parametrize('execution', ['main', 'thread', 'process'])
def test_func_favored(execution, session):
    if False:
        i = 10
        return i + 15
    session.parameters['arg'] = 'incorrect'
    task = FuncTask(run_parametrized_correct, start_cond=AlwaysTrue(), name='task', execution=execution, session=session)
    session.config.shut_cond = SchedulerCycles() == 1
    session.start()
    logger = task.logger
    assert 1 == logger.filter_by(action='run').count()
    assert 1 == logger.filter_by(action='success').count()

@pytest.mark.parametrize('execution', ['main', 'thread', 'process'])
def test_session_favored(execution, session):
    if False:
        for i in range(10):
            print('nop')
    session.parameters['arg'] = 'correct'
    task = FuncTask(run_parametrized, start_cond=AlwaysTrue(), name='task', execution=execution, session=session)
    session.config.shut_cond = SchedulerCycles() == 1
    session.start()
    logger = task.logger
    assert 1 == logger.filter_by(action='run').count()
    assert 1 == logger.filter_by(action='success').count()