from rocketry import Rocketry
from rocketry.conds import daily
app = Rocketry()

@app.task(daily)
def do_things():
    if False:
        print('Hello World!')
    ...
if __name__ == '__main__':
    app.run()