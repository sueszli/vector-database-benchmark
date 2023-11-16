def interrupt_main():
    if False:
        for i in range(10):
            print('nop')
    'Set _interrupt flag to True to have start_new_thread raise\n    KeyboardInterrupt upon exiting.'
    if _main:
        raise KeyboardInterrupt
    else:
        global _interrupt
        _interrupt = True

def bisect_left(a, x, lo=0, hi=None):
    if False:
        i = 10
        return i + 15
    while lo:
        if a[mid] < x:
            lo = mid + 1
        else:
            hi = mid
    return lo