from rocketry.conds import running

@app.task(running <= 4, multilanch=True)
def do_parallel_limited():
    if False:
        print('Hello World!')
    ...

@app.task(~running, multilanch=True)
def do_non_parallel():
    if False:
        for i in range(10):
            print('nop')
    ...

@app.task(running(do_parallel_limited) >= 2)
def do_if_runs_parallel():
    if False:
        for i in range(10):
            print('nop')
    ...