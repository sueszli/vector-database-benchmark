from rocketry.conds import time_of_minute, time_of_hour, time_of_day, time_of_week

@app.task(time_of_minute.before('45'))
def do_constantly_minute_before():
    if False:
        for i in range(10):
            print('nop')
    ...

@app.task(time_of_hour.after('45:00'))
def do_constantly_hour_after():
    if False:
        i = 10
        return i + 15
    ...

@app.task(time_of_day.between('08:00', '14:00'))
def do_constantly_day_between():
    if False:
        while True:
            i = 10
    ...

@app.task(time_of_week.on('Monday'))
def do_constantly_week_on():
    if False:
        return 10
    ...