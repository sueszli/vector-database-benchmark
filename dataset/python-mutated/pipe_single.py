from rocketry.conds import after_success, after_fail, after_finish

@app.task()
def do_things():
    if False:
        for i in range(10):
            print('nop')
    ...

@app.task(after_success(do_things))
def do_after_success():
    if False:
        i = 10
        return i + 15
    ...

@app.task(after_fail(do_things))
def do_after_fail():
    if False:
        return 10
    ...

@app.task(after_finish(do_things))
def do_after_fail_or_success():
    if False:
        return 10
    ...