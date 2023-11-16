from rocketry.conds import cron

@app.task(cron(minute='*/5'))
def do_simple():
    if False:
        return 10
    'Run at every 5th minute'
    ...

@app.task(cron(minute='*/2', hour='7-18', day_of_month='1,2,3', month='Feb-Aug/2'))
def do_complex():
    if False:
        print('Hello World!')
    'Run at:\n        - Every second minute\n        - Between 07:00 (7 a.m.) - 18:00 (6 p.m.)\n        - On 1st, 2nd and 3rd day of month\n        - From February to August every second month\n    '
    ...