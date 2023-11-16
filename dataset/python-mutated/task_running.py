from rocketry.conds import running

@app.task(end_cond=running.more_than('2 mins'))
def do_things():
    if False:
        return 10
    ...

@app.task(running(do_things))
def do_if_runs():
    if False:
        return 10
    ...

@app.task(running(do_things).less_than('2 mins'))
def do_if_runs_less_than():
    if False:
        i = 10
        return i + 15
    ...

@app.task(running(do_things).between('2 mins', '5 mins'))
def do_if_runs_between():
    if False:
        for i in range(10):
            print('nop')
    ...