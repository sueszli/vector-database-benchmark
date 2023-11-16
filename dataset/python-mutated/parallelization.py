@app.task(execution='main')
def do_unparallel():
    if False:
        for i in range(10):
            print('nop')
    ...

@app.task(execution='async')
async def do_unparallel():
    ...

@app.task(execution='thread')
def do_on_separate_thread():
    if False:
        for i in range(10):
            print('nop')
    ...

@app.task(execution='process')
def do_on_separate_process():
    if False:
        print('Hello World!')
    ...