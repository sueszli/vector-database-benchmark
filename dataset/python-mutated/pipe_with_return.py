from rocketry.conds import daily, after_success
from rocketry.args import Return

@app.task(daily)
def do_first():
    if False:
        while True:
            i = 10
    return 'Hello World'

@app.task(after_success(do_first))
def do_second(arg=Return(do_first)):
    if False:
        for i in range(10):
            print('nop')
    ...