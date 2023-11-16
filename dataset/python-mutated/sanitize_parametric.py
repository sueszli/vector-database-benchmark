def a_source_x():
    if False:
        return 10
    ...

def a_source_y():
    if False:
        print('Hello World!')
    ...

def a_sink_x(x):
    if False:
        print('Hello World!')
    ...

def a_sink_y(x):
    if False:
        print('Hello World!')
    ...

def sanitize_a_source_tito(x):
    if False:
        print('Hello World!')
    return x

def sanitize_a_sink_tito(x):
    if False:
        print('Hello World!')
    return x

def partial_issue_sources():
    if False:
        for i in range(10):
            print('nop')
    if 1 > 2:
        x = a_source_x()
        return sanitize_a_sink_tito(x)
    else:
        return a_source_y()

def partial_issue_sinks(x):
    if False:
        print('Hello World!')
    if 1 > 2:
        a_sink_x(x)
    else:
        y = sanitize_a_source_tito(x)
        a_sink_y(y)

def partial_issue_sanitize():
    if False:
        return 10
    x = partial_issue_sources()
    partial_issue_sinks(x)