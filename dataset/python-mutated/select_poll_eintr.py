try:
    import time, gc, select, socket, _thread
    time.time_ns
    select.poll
except (ImportError, AttributeError):
    print('SKIP')
    raise SystemExit

def thread_main():
    if False:
        return 10
    lock.acquire()
    time.sleep(0.2)
    print('thread gc start')
    gc.collect()
    print('thread gc end')
lock = _thread.allocate_lock()
lock.acquire()
_thread.start_new_thread(thread_main, ())
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s.bind(socket.getaddrinfo('127.0.0.1', 8000)[0][-1])
poller = select.poll()
poller.register(s, select.POLLIN)
print('poll')
lock.release()
t0 = time.time_ns()
result = poller.poll(400)
dt_ms = (time.time_ns() - t0) / 1000000.0
print('result:', result)
if 380 <= dt_ms <= 600:
    print('dt in range')
else:
    print('dt not in range:', dt_ms)
s.close()