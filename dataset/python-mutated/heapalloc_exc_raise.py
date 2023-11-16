import micropython
e = ValueError('error')

def func():
    if False:
        return 10
    micropython.heap_lock()
    try:
        raise e
    except Exception as e2:
        print(e2)
    micropython.heap_unlock()
func()
print('ok')