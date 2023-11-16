from rocketry.conds import minutely, hourly, daily, weekly, monthly

@app.task(minutely.before('45'))
def do_before():
    if False:
        while True:
            i = 10
    ...

@app.task(hourly.after('45:00'))
def do_after():
    if False:
        return 10
    ...

@app.task(daily.between('08:00', '14:00'))
def do_between():
    if False:
        while True:
            i = 10
    ...

@app.task(daily.at('11:00'))
def do_at():
    if False:
        print('Hello World!')
    ...

@app.task(weekly.on('Monday'))
def do_on():
    if False:
        i = 10
        return i + 15
    ...

@app.task(monthly.starting('3rd'))
def do_starting():
    if False:
        print('Hello World!')
    ...