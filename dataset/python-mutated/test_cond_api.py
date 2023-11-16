import logging
from rocketry import Rocketry
from rocketry.args.builtin import Return, Task
from rocketry.conditions import TaskStarted
from rocketry.conds import false, true, daily, time_of_hour, after_fail, after_success

def set_logging_defaults():
    if False:
        while True:
            i = 10
    task_logger = logging.getLogger('rocketry.task')
    task_logger.handlers = []
    task_logger.setLevel(logging.WARNING)

def test_app_run():
    if False:
        print('Hello World!')
    set_logging_defaults()
    app = Rocketry(execution='main')

    @app.task(false)
    def do_never():
        if False:
            return 10
        ...

    @app.task(daily)
    def do_daily():
        if False:
            while True:
                i = 10
        ...

    @app.task(after_success(do_daily))
    def do_after():
        if False:
            while True:
                i = 10
        ...

    @app.task(daily & (after_fail(do_never) | time_of_hour.before('10:00') | after_success('do_daily')))
    def do_daily_complex():
        if False:
            print('Hello World!')
        ...
    app.session.config.shut_cond = TaskStarted(task='do_after')
    app.run()
    logger = app.session['do_after'].logger
    assert logger.filter_by(action='success').count() == 1

def test_pipe():
    if False:
        print('Hello World!')
    set_logging_defaults()
    app = Rocketry(execution='main')

    @app.task(true)
    def do_first():
        if False:
            while True:
                i = 10
        return 'hello'

    @app.task(after_success(do_first))
    def do_second(arg=Return(do_first)):
        if False:
            for i in range(10):
                print('nop')
        assert arg == 'hello'
    app.session.config.shut_cond = TaskStarted(task=do_second)
    app.run()
    logger = app.session['do_second'].logger
    assert logger.filter_by(action='success').count() == 1

def test_custom_cond():
    if False:
        for i in range(10):
            print('nop')
    set_logging_defaults()
    app = Rocketry(execution='main')

    @app.cond('is foo')
    def is_foo(task=Task()):
        if False:
            print('Hello World!')
        assert task.name == 'do_things'
        return True

    @app.task(true & is_foo)
    def do_things():
        if False:
            while True:
                i = 10
        ...
    app.session.config.shut_cond = TaskStarted(task=do_things)
    app.run()
    logger = app.session['do_things'].logger
    assert logger.filter_by(action='success').count() == 1

def test_custom_cond_parametrized():
    if False:
        return 10
    set_logging_defaults()
    app = Rocketry(execution='main')

    @app.cond('is foo')
    def is_foo(x, task=Task()):
        if False:
            while True:
                i = 10
        assert x == 'a value'
        assert task.name == 'do_things'
        return True

    @app.task(true & is_foo(x='a value'))
    def do_things():
        if False:
            return 10
        ...
    assert is_foo(x='a value') is not is_foo(x='a value')
    app.session.config.shut_cond = TaskStarted(task=do_things)
    app.run()
    logger = app.session['do_things'].logger
    assert logger.filter_by(action='success').count() == 1