from rocketry.conds import every

@app.task(every('10 seconds'))
def do_constantly():
    if False:
        for i in range(10):
            print('nop')
    ...

@app.task(every('1 minute'))
def do_minutely():
    if False:
        return 10
    ...

@app.task(every('1 hour'))
def do_hourly():
    if False:
        while True:
            i = 10
    ...

@app.task(every('1 day'))
def do_daily():
    if False:
        while True:
            i = 10
    ...

@app.task(every('2 days 2 hours 20 seconds'))
def do_custom():
    if False:
        print('Hello World!')
    ...