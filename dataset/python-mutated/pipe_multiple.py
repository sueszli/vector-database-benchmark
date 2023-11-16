from rocketry.conds import after_all_success, after_any_success, after_any_finish, after_any_fail

@app.task()
def do_a():
    if False:
        while True:
            i = 10
    ...

@app.task()
def do_b():
    if False:
        print('Hello World!')
    ...

@app.task(after_all_success(do_a, do_b))
def do_all_succeeded():
    if False:
        for i in range(10):
            print('nop')
    ...

@app.task(after_any_success(do_a, do_b))
def do_any_succeeded():
    if False:
        print('Hello World!')
    ...

@app.task(after_any_fail(do_a, do_b))
def do_any_failed():
    if False:
        print('Hello World!')
    ...

@app.task(after_any_finish(do_a, do_b))
def do_any_failed_or_succeeded():
    if False:
        i = 10
        return i + 15
    ...