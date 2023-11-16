import time
import _thread

def thread_entry():
    if False:
        i = 10
        return i + 15
    _thread.exit()
for i in range(2):
    while True:
        try:
            _thread.start_new_thread(thread_entry, ())
            break
        except OSError:
            pass
time.sleep(1)
print('done')