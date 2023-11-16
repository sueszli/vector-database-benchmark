def a_source():
    if False:
        return 10
    ...

def a_sink(x):
    if False:
        print('Hello World!')
    ...

def b_sink(x):
    if False:
        while True:
            i = 10
    ...

def sanitize_a_source_tito(x):
    if False:
        return 10
    return x

def sanitize_a_sink_tito(x):
    if False:
        while True:
            i = 10
    return x

def sanitize_b_sink_tito(x):
    if False:
        while True:
            i = 10
    return x

def test_source_a_sanitize_a_kept():
    if False:
        print('Hello World!')
    return sanitize_a_sink_tito(a_source())

def test_source_a_sanitize_a_b_discarded():
    if False:
        return 10
    return sanitize_b_sink_tito(sanitize_a_sink_tito(a_source()))

def test_sink_a_sanitize_a_discarded(x):
    if False:
        i = 10
        return i + 15
    a_sink(sanitize_a_source_tito(x))