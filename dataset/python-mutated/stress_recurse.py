import _thread

def foo():
    if False:
        return 10
    foo()

def thread_entry():
    if False:
        print('Hello World!')
    try:
        foo()
    except RuntimeError:
        print('RuntimeError')
    global finished
    finished = True
finished = False
_thread.start_new_thread(thread_entry, ())
while not finished:
    pass
print('done')