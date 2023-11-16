from rocketry.conds import crontime

@app.task(crontime('* * * * *'))
def do_minutely():
    if False:
        while True:
            i = 10
    ...

@app.task(crontime('*/2 12-18 * Oct Fri'))
def do_complex():
    if False:
        for i in range(10):
            print('nop')
    'Run at every 2nd minute past every hour from 12 through 18 on Friday in October.'
    ...