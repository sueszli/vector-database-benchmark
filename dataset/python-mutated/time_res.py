try:
    import time
except ImportError:
    print('SKIP')
    raise SystemExit

def gmtime_time():
    if False:
        while True:
            i = 10
    return time.gmtime(time.time())

def localtime_time():
    if False:
        for i in range(10):
            print('nop')
    return time.localtime(time.time())

def test():
    if False:
        while True:
            i = 10
    TEST_TIME = 2500
    EXPECTED_MAP = (('time', 3), ('gmtime', 3), ('localtime', 3), ('gmtime_time', 3), ('localtime_time', 3), ('ticks_ms', 15), ('ticks_us', 15), ('ticks_ns', 15), ('ticks_cpu', 15))
    results_map = {}
    end_time = time.ticks_ms() + TEST_TIME
    while time.ticks_diff(end_time, time.ticks_ms()) > 0:
        time.sleep_ms(100)
        for (func_name, _) in EXPECTED_MAP:
            try:
                time_func = getattr(time, func_name, None) or globals()[func_name]
                now = time_func()
            except (KeyError, AttributeError):
                continue
            try:
                results_map[func_name].add(now)
            except KeyError:
                results_map[func_name] = {now}
    for (func_name, min_len) in EXPECTED_MAP:
        print('Testing %s' % func_name)
        results = results_map.get(func_name)
        if results is None:
            pass
        elif func_name == 'ticks_cpu' and results == {0}:
            pass
        elif len(results) < min_len:
            print('%s() returns %s result%s in %s ms, expecting >= %s' % (func_name, len(results), 's'[:len(results) != 1], TEST_TIME, min_len))
test()