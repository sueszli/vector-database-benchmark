from pathlib import Path
from rocketry import Rocketry
from rocketry.conds import daily
app = Rocketry()

@app.cond()
def file_exists(file):
    if False:
        for i in range(10):
            print('nop')
    'Custom condition that checks if file exists'
    return Path(file).exists()

@app.task(daily & file_exists('data.csv'))
def do_things():
    if False:
        return 10
    'Task that runs once a day when data.csv exists'
    ...
if __name__ == '__main__':
    app.run()