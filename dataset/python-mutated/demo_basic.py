from rocketry import Rocketry
app = Rocketry()

@app.task('daily')
def do_things():
    if False:
        i = 10
        return i + 15
    ...

@app.task("after task 'do_things'")
def do_after_things():
    if False:
        return 10
    ...
if __name__ == '__main__':
    app.run()