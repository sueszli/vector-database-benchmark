from rocketry.conds import crontime

@app.task(crontime(minute='*/5'))
def do_simple():
    if False:
        for i in range(10):
            print('nop')
    'Run at every 5th minute'
    ...

@app.task(crontime(minute='*/2', hour='7-18', day_of_month='1,2,3', month='Feb-Aug/2'))
def do_complex():
    if False:
        while True:
            i = 10
    'Run at:\n        - Every second minute\n        - Between 07:00 (7 a.m.) - 18:00 (6 p.m.)\n        - On 1st, 2nd and 3rd day of month\n        - From February to August every second month\n    '
    ...