from rocketry import Rocketry
from rocketry.args import Return, Session, Arg, FuncArg
from rocketry.conds import daily, time_of_week, after_success
app = Rocketry()

@app.cond()
def is_foo():
    if False:
        while True:
            i = 10
    'This is a custom condition'
    ...
    return True

@app.task(daily & is_foo)
def do_daily():
    if False:
        while True:
            i = 10
    'This task runs once a day when foo is true'
    ...
    return ...

@app.task((daily.at('10:00') | daily.at('19:00')) & time_of_week.between('Mon', 'Fri'), execution='process')
def do_complex():
    if False:
        while True:
            i = 10
    'This task runs on complex interval and on separate process'
    ...
    return ...

@app.task(after_success(do_daily))
def do_after_another(arg=Return(do_daily)):
    if False:
        while True:
            i = 10
    "This task runs after 'do_daily' and it has its the\n    return argument as an input"
    ...

@app.task(daily)
def do_with_params(arg1=FuncArg(lambda : ...), arg2=Arg('myparam')):
    if False:
        for i in range(10):
            print('nop')
    'This task runs with variety of arguments'
    ...

@app.task(daily, execution='thread')
def do_on_session(session=Session()):
    if False:
        i = 10
        return i + 15
    'This task modifies the scheduling session'
    for task in session.tasks:
        if task.name == 'do_after_another':
            task.run(arg='...')
    session.shut_down()
if __name__ == '__main__':
    app.params(myparam='...')
    app.run()