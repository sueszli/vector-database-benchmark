import time
import _thread

def thread_entry():
    if False:
        for i in range(10):
            print('nop')
    raise SystemExit
for i in range(2):
    while True:
        try:
            _thread.start_new_thread(thread_entry, ())
            break
        except OSError:
            pass
time.sleep(1)
print('done')