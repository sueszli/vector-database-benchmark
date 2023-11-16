from builtins import _test_sink

def only_applies_to_first():
    if False:
        while True:
            i = 10
    return (1, 0)

def only_applies_to_second():
    if False:
        while True:
            i = 10
    return (0, 1)

def only_applies_to_nested():
    if False:
        while True:
            i = 10
    return ((0, 1), (0, 0))

def issue_only_with_first():
    if False:
        return 10
    (issue, no_issue) = only_applies_to_first()
    _test_sink(issue)
    _test_sink(no_issue)

def issue_only_with_second():
    if False:
        while True:
            i = 10
    (no_issue, issue) = only_applies_to_second()
    _test_sink(no_issue)
    _test_sink(issue)

def issue_only_with_nested_first():
    if False:
        return 10
    (first, second) = only_applies_to_nested()
    (a, issue) = first
    (c, d) = second
    _test_sink(issue)
    _test_sink(a)
    _test_sink(c)
    _test_sink(d)
    return only_applies_to_nested()

def only_applies_to_a_key():
    if False:
        print('Hello World!')
    return {'a': 1}

def issue_only_with_a_key():
    if False:
        return 10
    d = only_applies_to_a_key()
    _test_sink(d['a'])
    _test_sink(d['b'])

def only_applies_to_a_member():
    if False:
        for i in range(10):
            print('nop')
    ...

def issue_with_member():
    if False:
        return 10
    x = only_applies_to_a_member()
    _test_sink(x.a)
    _test_sink(x.b)

def tito(x):
    if False:
        print('Hello World!')
    return