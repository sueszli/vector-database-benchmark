from builtins import _cookies, _rce, _sql, _test_sink, _test_source, _user_controlled
from typing import Sequence, TypeVar
T = TypeVar('T')

class C_sanitized_a_source:
    attribute = None

    def __init__(self, value):
        if False:
            for i in range(10):
                print('nop')
        self.instance = value
        self.attribute = value

class C_sanitized_b_source:
    attribute = None

    def __init__(self, value):
        if False:
            while True:
                i = 10
        self.instance = value
        self.attribute = value

class C_sanitized_ab_sources:
    attribute = None

    def __init__(self, value):
        if False:
            for i in range(10):
                print('nop')
        self.instance = value
        self.attribute = value

class C_sanitized_all_sources:
    attribute = None

    def __init__(self, value):
        if False:
            for i in range(10):
                print('nop')
        self.instance = value
        self.attribute = value

class C_sanitized_a_sink:
    attribute = ...

    def __init__(self, value):
        if False:
            print('Hello World!')
        self.instance = value
        self.attribute = value

class C_sanitized_b_sink:
    attribute = ...

    def __init__(self, value):
        if False:
            for i in range(10):
                print('nop')
        self.instance = value
        self.attribute = value

class C_sanitized_ab_sinks:
    attribute = ...

    def __init__(self, value):
        if False:
            return 10
        self.instance = value
        self.attribute = value

class C_sanitized_all_sinks:
    attribute = ...

    def __init__(self, value):
        if False:
            print('Hello World!')
        self.instance = value
        self.attribute = value

def return_taint_sanitize(arg: T) -> T:
    if False:
        return 10
    "Identity function that returns the argument unmodified, but is marked as\n    'Sanitize' in the accompanying .pysa file\n    "
    return arg

def test1():
    if False:
        return 10
    tainted = object()
    tainted.id = _test_source()
    test2(tainted)
    test3(tainted)

def test2(tainted_other):
    if False:
        while True:
            i = 10
    tainted = return_taint_sanitize(tainted_other)
    _test_sink(tainted.id)

def test3(colliding_name):
    if False:
        while True:
            i = 10
    colliding_name = return_taint_sanitize(colliding_name)
    _test_sink(colliding_name.id)

def source_with_tito(x):
    if False:
        for i in range(10):
            print('nop')
    return x

def sanitize_sources(x):
    if False:
        while True:
            i = 10
    _test_sink(x)
    return source_with_tito(x)

def sanitize_sinks(x):
    if False:
        for i in range(10):
            print('nop')
    _test_sink(x)
    return source_with_tito(x)

def sanitize_tito(x):
    if False:
        for i in range(10):
            print('nop')
    _test_sink(x)
    return source_with_tito(x)

def a_source():
    if False:
        return 10
    ...

def b_source():
    if False:
        return 10
    ...

def sanitize_test_a_source():
    if False:
        for i in range(10):
            print('nop')
    if 1 > 2:
        x = a_source()
    else:
        x = b_source()
    return x

def sanitize_test_b_source():
    if False:
        print('Hello World!')
    if 1 > 2:
        x = a_source()
    else:
        x = b_source()
    return x

def sanitize_a_and_b_source():
    if False:
        for i in range(10):
            print('nop')
    if 1 > 2:
        x = a_source()
    else:
        x = b_source()
    return x

def a_sink(x):
    if False:
        print('Hello World!')
    ...

def b_sink(x):
    if False:
        i = 10
        return i + 15
    ...

def sanitize_a_sink(x):
    if False:
        i = 10
        return i + 15
    if 1 > 2:
        a_sink(x)
    else:
        b_sink(x)

def sanitize_b_sink(x):
    if False:
        while True:
            i = 10
    if 1 > 2:
        a_sink(x)
    else:
        b_sink(x)

def sanitize_a_and_b_sinks(x):
    if False:
        return 10
    if 1 > 2:
        a_sink(x)
    else:
        b_sink(x)

def sanitize_a_source_tito(x):
    if False:
        i = 10
        return i + 15
    return x

def no_propagation_with_sanitize_a_source_tito():
    if False:
        print('Hello World!')
    a = a_source()
    b = sanitize_a_source_tito(a)
    return b

def propagation_of_b_with_sanitize_a_source_tito():
    if False:
        i = 10
        return i + 15
    b = b_source()
    tito = sanitize_a_source_tito(b)
    return tito

def propagation_of_sanitize_a_source_tito(x):
    if False:
        for i in range(10):
            print('nop')
    return sanitize_a_source_tito(x)

def no_issue_through_propagation_of_sanitize_a_source_tito():
    if False:
        while True:
            i = 10
    x = a_source()
    y = propagation_of_sanitize_a_source_tito(x)
    a_sink(y)

def propagation_of_sanitize_a_source_in_sink_trace(x):
    if False:
        while True:
            i = 10
    y = propagation_of_sanitize_a_source_tito(x)
    a_sink(y)

def no_issue_propagation_of_sanitize_a_source_in_sink_trace():
    if False:
        while True:
            i = 10
    x = a_source()
    propagation_of_sanitize_a_source_in_sink_trace(x)

def sanitize_b_source_tito(x):
    if False:
        return 10
    return x

def sanitize_test_source_tito(x):
    if False:
        for i in range(10):
            print('nop')
    return x

def combine_sanitize_a_source_b_source_in_sink_trace(x):
    if False:
        i = 10
        return i + 15
    y = sanitize_b_source_tito(x)
    propagation_of_sanitize_a_source_in_sink_trace(y)

def sanitize_a_sink_tito(x):
    if False:
        i = 10
        return i + 15
    return x

def no_propagation_of_a_sink(x):
    if False:
        print('Hello World!')
    y = sanitize_a_sink_tito(x)
    a_sink(y)

def propagation_of_b_sink(x):
    if False:
        while True:
            i = 10
    y = sanitize_a_sink_tito(x)
    b_sink(y)

def combine_sanitize_a_source_a_sink_tito(x):
    if False:
        return 10
    y = sanitize_a_source_tito(x)
    z = sanitize_a_sink_tito(y)
    return z

def no_issue_through_combine_sanitize_a_source_a_sink_tito():
    if False:
        print('Hello World!')
    x = a_source()
    y = combine_sanitize_a_source_a_sink_tito(x)
    a_sink(y)

def propagation_of_sanitize_a_sink_in_source_trace():
    if False:
        for i in range(10):
            print('nop')
    x = a_source()
    y = sanitize_a_sink_tito(x)
    return y

def no_issue_propagation_of_sanitize_a_sink_in_source_trace():
    if False:
        i = 10
        return i + 15
    x = propagation_of_sanitize_a_sink_in_source_trace()
    a_sink(x)

def sanitize_b_sink_tito(x):
    if False:
        while True:
            i = 10
    return x

def combine_sanitize_a_sink_b_sink_in_source_trace():
    if False:
        while True:
            i = 10
    x = propagation_of_sanitize_a_sink_in_source_trace()
    y = sanitize_b_sink_tito(x)
    return y

def sanitize_a_source_tito_with_sink(x):
    if False:
        while True:
            i = 10
    a_sink(x)
    return x

def sanitize_with_user_declared_source():
    if False:
        for i in range(10):
            print('nop')
    return 0

def sanitize_with_user_declared_sink(x):
    if False:
        return 10
    return

def test4():
    if False:
        while True:
            i = 10
    x = a_source()
    y = sanitize_a_source_tito_with_sink(x)
    a_sink(y)

def sanitize_b_sink_tito(x):
    if False:
        while True:
            i = 10
    return x

def no_issue_fixpoint_sanitize_sources():
    if False:
        print('Hello World!')
    if 1 > 2:
        x = a_source()
        return sanitize_a_sink_tito(x)
    else:
        x = _test_source()
        y = sanitize_a_sink_tito(x)
        return sanitize_b_sink_tito(y)

def no_issue_fixpoint_sanitize_sinks(x):
    if False:
        while True:
            i = 10
    if 1 > 2:
        a_sink(x)
    else:
        y = sanitize_a_source_tito(x)
        b_sink(y)

def no_issue_fixpoint_sanitize():
    if False:
        return 10
    x = no_issue_fixpoint_sanitize_sources()
    no_issue_fixpoint_sanitize_sinks(x)

def partial_issue_sources():
    if False:
        i = 10
        return i + 15
    if 1 > 2:
        x = a_source()
        return sanitize_a_sink_tito(x)
    else:
        return a_source()

def partial_issue_sinks(x):
    if False:
        while True:
            i = 10
    if 1 > 2:
        a_sink(x)
    else:
        y = sanitize_a_source_tito(x)
        a_sink(y)

def partial_issue_sanitize():
    if False:
        return 10
    x = partial_issue_sources()
    partial_issue_sinks(x)

def sanitize_test_a_source_attribute():
    if False:
        while True:
            i = 10
    if 1 > 2:
        x = a_source()
    else:
        x = b_source()
    c = C_sanitized_a_source(x)
    _test_sink(c.attribute)

def sanitize_test_a_source_attribute_in_sink_trace(x):
    if False:
        print('Hello World!')
    c = C_sanitized_a_source(x)
    _test_sink(c.attribute)

def no_issue_sanitize_test_a_source_attribute_in_sink_trace():
    if False:
        print('Hello World!')
    x = a_source()
    sanitize_test_a_source_attribute_in_sink_trace(x)

def issue_sanitize_test_a_source_attribute_in_sink_trace():
    if False:
        return 10
    x = b_source()
    sanitize_test_a_source_attribute_in_sink_trace(x)

def sanitize_test_a_source_attribute_in_tito(x):
    if False:
        while True:
            i = 10
    c = C_sanitized_a_source(x)
    return c.attribute

def sanitize_test_b_source_attribute():
    if False:
        while True:
            i = 10
    if 1 > 2:
        x = a_source()
    else:
        x = b_source()
    c = C_sanitized_b_source(x)
    _test_sink(c.attribute)

def sanitize_test_ab_sources_attribute():
    if False:
        return 10
    if 1 > 2:
        x = a_source()
    else:
        x = b_source()
    c = C_sanitized_ab_sources(x)
    _test_sink(c.attribute)

def sanitize_test_all_sources_attribute():
    if False:
        while True:
            i = 10
    if 1 > 2:
        x = a_source()
    elif 2 > 3:
        x = b_source()
    else:
        x = _test_source()
    c = C_sanitized_all_sources(x)
    _test_sink(c.attribute)

def sanitize_test_a_source_instance():
    if False:
        print('Hello World!')
    if 1 > 2:
        x = a_source()
    else:
        x = b_source()
    c = C_sanitized_a_source(x)
    _test_sink(c.instance)

def sanitize_test_b_source_instance():
    if False:
        return 10
    if 1 > 2:
        x = a_source()
    else:
        x = b_source()
    c = C_sanitized_b_source(x)
    _test_sink(c.instance)

def sanitize_test_ab_sources_instance():
    if False:
        return 10
    if 1 > 2:
        x = a_source()
    else:
        x = b_source()
    c = C_sanitized_ab_sources(x)
    _test_sink(c.instance)

def sanitize_test_all_sources_instance():
    if False:
        i = 10
        return i + 15
    if 1 > 2:
        x = a_source()
    elif 2 > 3:
        x = b_source()
    else:
        x = _test_source()
    c = C_sanitized_all_sources(x)
    _test_sink(c.instance)

def sanitize_a_sink_attribute(c: C_sanitized_a_sink):
    if False:
        for i in range(10):
            print('nop')
    if 1 > 2:
        a_sink(c.attribute)
    else:
        b_sink(c.attribute)

def sanitize_a_sink_attribute_in_source_trace():
    if False:
        for i in range(10):
            print('nop')
    x = a_source()
    y = C_sanitized_a_sink(x)
    return y.attribute

def no_issue_sanitize_a_sink_attribute_in_source_trace():
    if False:
        print('Hello World!')
    x = sanitize_a_sink_attribute_in_source_trace()
    a_sink(x)

def issue_sanitize_a_sink_attribute_in_source_trace():
    if False:
        while True:
            i = 10
    x = sanitize_a_sink_attribute_in_source_trace()
    b_sink(x)

def sanitize_b_sink_attribute(c: C_sanitized_b_sink):
    if False:
        return 10
    if 1 > 2:
        a_sink(c.attribute)
    else:
        b_sink(c.attribute)

def sanitize_ab_sinks_attribute(c: C_sanitized_ab_sinks):
    if False:
        return 10
    if 1 > 2:
        a_sink(c.attribute)
    else:
        b_sink(c.attribute)

def sanitize_all_sinks_attribute(c: C_sanitized_all_sinks):
    if False:
        while True:
            i = 10
    if 1 > 2:
        a_sink(c.attribute)
    elif 2 > 3:
        b_sink(c.attribute)
    else:
        _test_sink(c.attribute)

def sanitize_a_sink_instance(c: C_sanitized_a_sink):
    if False:
        print('Hello World!')
    if 1 > 2:
        a_sink(c.instance)
    else:
        b_sink(c.instance)

def sanitize_b_sink_instance(c: C_sanitized_b_sink):
    if False:
        i = 10
        return i + 15
    if 1 > 2:
        a_sink(c.instance)
    else:
        b_sink(c.instance)

def sanitize_ab_sinks_instance(c: C_sanitized_ab_sinks):
    if False:
        for i in range(10):
            print('nop')
    if 1 > 2:
        a_sink(c.instance)
    else:
        b_sink(c.instance)

def sanitize_all_sinks_instance(c: C_sanitized_all_sinks):
    if False:
        while True:
            i = 10
    if 1 > 2:
        a_sink(c.instance)
    elif 2 > 3:
        b_sink(c.instance)
    else:
        _test_sink(c.instance)

def sanitize_test_a_sink_attribute():
    if False:
        while True:
            i = 10
    sanitize_a_sink_attribute(_test_source())

def sanitize_test_b_sink_attribute():
    if False:
        print('Hello World!')
    sanitize_b_sink_attribute(_test_source())

def sanitize_test_ab_sinks_attribute():
    if False:
        return 10
    sanitize_ab_sinks_attribute(_test_source())

def sanitize_test_all_sinks_attribute():
    if False:
        while True:
            i = 10
    sanitize_all_sinks_attribute(_test_source())
    c = C_sanitized_all_sinks({})
    c.attribute = _test_source()

def sanitize_test_a_sink_instance():
    if False:
        for i in range(10):
            print('nop')
    sanitize_a_sink_instance(_test_source())

def sanitize_test_b_sink_instance():
    if False:
        print('Hello World!')
    sanitize_b_sink_instance(_test_source())

def sanitize_test_ab_sinks_instance():
    if False:
        return 10
    sanitize_ab_sinks_instance(_test_source())

def sanitize_test_all_sinks_instance():
    if False:
        i = 10
        return i + 15
    sanitize_all_sinks_instance(_test_source())
    c = C_sanitized_all_sinks({})
    c.instance = _test_source()

def sanitize_parameter(x, y):
    if False:
        i = 10
        return i + 15
    _test_sink(x)
    _test_sink(y)
    return source_with_tito(x) + source_with_tito(y)

def sanitize_parameter_all_tito(x, y):
    if False:
        i = 10
        return i + 15
    _test_sink(x)
    _test_sink(y)
    return source_with_tito(x) + source_with_tito(y)

def sanitize_parameter_no_user_controlled(x, y):
    if False:
        for i in range(10):
            print('nop')
    if 1 > 2:
        return x
    elif 2 > 3:
        return y
    elif 3 > 4:
        _sql(x)
    else:
        _rce(y)

def propagation_of_sanitize_parameter_no_user_controlled(a, b):
    if False:
        return 10
    sanitize_parameter_no_user_controlled(b, a)

def no_issue_propagation_of_sanitize_parameter_no_user_controlled():
    if False:
        for i in range(10):
            print('nop')
    x = _user_controlled()
    propagation_of_sanitize_parameter_no_user_controlled(0, x)

def issue_propagation_of_sanitize_parameter_no_user_controlled():
    if False:
        i = 10
        return i + 15
    x = _cookies()
    propagation_of_sanitize_parameter_no_user_controlled(0, x)

def sanitize_parameter_no_sql(x):
    if False:
        while True:
            i = 10
    if 1 > 2:
        _sql(x)
    elif 2 > 3:
        _rce(x)
    else:
        return x

def sanitize_parameter_no_rce(x):
    if False:
        while True:
            i = 10
    if 1 > 2:
        _sql(x)
    elif 2 > 3:
        _rce(x)
    else:
        return x

def sanitize_parameter_no_user_controlled_tito(x, y):
    if False:
        while True:
            i = 10
    if 1 > 2:
        return x
    else:
        return y

def no_propagation_with_sanitize_parameter_no_user_controlled_tito():
    if False:
        i = 10
        return i + 15
    a = _user_controlled()
    b = sanitize_parameter_no_user_controlled_tito(a, 0)
    return b

def propagation_of_cookies_with_sanitize_parameter_no_user_controlled_tito():
    if False:
        for i in range(10):
            print('nop')
    b = _cookies()
    tito = sanitize_parameter_no_user_controlled_tito(b, 0)
    return tito

def propagation_of_sanitize_parameter_no_user_controlled_tito(a, b):
    if False:
        print('Hello World!')
    return sanitize_parameter_no_user_controlled_tito(b, a)

def propagation_of_sanitize_parameter_no_user_controlled_tito_in_sink_trace(x):
    if False:
        i = 10
        return i + 15
    y = propagation_of_sanitize_parameter_no_user_controlled_tito(0, x)
    _sql(y)

def no_issue_propagation_of_sanitize_parameter_no_user_controlled_tito_in_sink_trace():
    if False:
        for i in range(10):
            print('nop')
    x = _user_controlled()
    propagation_of_sanitize_parameter_no_user_controlled_tito_in_sink_trace(x)

def issue_propagation_of_sanitize_parameter_no_user_controlled_tito_in_sink_trace():
    if False:
        while True:
            i = 10
    x = _cookies()
    propagation_of_sanitize_parameter_no_user_controlled_tito_in_sink_trace(x)

def sanitize_parameter_no_sql_tito(x, y):
    if False:
        while True:
            i = 10
    if 1 > 2:
        return x
    else:
        return y

def no_propagation_with_sanitize_parameter_no_sql_tito(x):
    if False:
        while True:
            i = 10
    y = sanitize_parameter_no_sql_tito(x, 0)
    _sql(y)

def propagation_of_rce_with_sanitize_parameter_no_sql_tito(x):
    if False:
        i = 10
        return i + 15
    y = sanitize_parameter_no_sql_tito(x, 0)
    _rce(y)

def propagation_of_sanitize_parameter_no_sql_tito(a, b):
    if False:
        for i in range(10):
            print('nop')
    return sanitize_parameter_no_sql_tito(b, a)

def propagation_of_sanitize_parameter_no_sql_tito_in_source_trace():
    if False:
        return 10
    x = _user_controlled()
    return propagation_of_sanitize_parameter_no_sql_tito(0, x)

def no_issue_propagation_of_sanitize_parameter_no_sql_tito_in_source_trace():
    if False:
        i = 10
        return i + 15
    x = propagation_of_sanitize_parameter_no_sql_tito_in_source_trace()
    _sql(x)

def issue_propagation_of_sanitize_parameter_no_sql_tito_in_source_trace():
    if False:
        while True:
            i = 10
    x = propagation_of_sanitize_parameter_no_sql_tito_in_source_trace()
    _rce(x)

def sanitize_parameter_with_user_declared_sink(x):
    if False:
        while True:
            i = 10
    return

def sanitize_return(x):
    if False:
        i = 10
        return i + 15
    _test_sink(x)
    return source_with_tito(x)

def sanitize_return_no_user_controlled(x):
    if False:
        print('Hello World!')
    if 1 > 2:
        return _user_controlled()
    elif 2 > 3:
        return _cookies()
    else:
        return x

def sanitize_return_no_sql(x):
    if False:
        i = 10
        return i + 15
    return x

def propagation_of_sanitize_return_no_sql(x):
    if False:
        print('Hello World!')
    return sanitize_return_no_sql(x)

def propagation_of_sanitize_return_no_sql_in_source_trace():
    if False:
        return 10
    x = _user_controlled()
    y = propagation_of_sanitize_return_no_sql(x)
    return y

def no_issue_propagation_of_sanitize_return_no_sql_in_source_trace():
    if False:
        return 10
    x = propagation_of_sanitize_return_no_sql_in_source_trace()
    _sql(x)

def issue_propagation_of_sanitize_return_no_sql_in_source_trace():
    if False:
        while True:
            i = 10
    x = propagation_of_sanitize_return_no_sql_in_source_trace()
    _rce(x)

def sanitize_return_no_cookies():
    if False:
        i = 10
        return i + 15
    if 1 > 2:
        x = _user_controlled()
    else:
        x = _cookies()
    return x

def sanitize_return_no_user_controlled_cookies():
    if False:
        print('Hello World!')
    if 1 > 2:
        x = _user_controlled()
    else:
        x = _cookies()
    return x

def sanitize_return_no_rce():
    if False:
        return 10
    return _user_controlled()

def propagation_of_sanitize_return_no_rce():
    if False:
        print('Hello World!')
    return sanitize_return_no_rce()

def no_issue_propagation_of_sanitize_return_no_rce():
    if False:
        i = 10
        return i + 15
    x = propagation_of_sanitize_return_no_rce()
    _rce(x)

def issue_propagation_of_sanitize_return_no_rce():
    if False:
        while True:
            i = 10
    x = propagation_of_sanitize_return_no_rce()
    _sql(x)

def sanitize_return_with_user_declared_source(x):
    if False:
        return 10
    return 0

def sanitize_all_parameters(x, y):
    if False:
        while True:
            i = 10
    _test_sink(x)
    _test_sink(y)
    return source_with_tito(x) + source_with_tito(y)

def sanitize_all_parameters_all_tito(x, y):
    if False:
        i = 10
        return i + 15
    _test_sink(x)
    _test_sink(y)
    return source_with_tito(x) + source_with_tito(y)

def sanitize_all_parameters_no_user_controlled(x):
    if False:
        return 10
    _test_sink(x)
    return x

def propagation_of_sanitize_all_parameters_no_user_controlled(x):
    if False:
        while True:
            i = 10
    sanitize_all_parameters_no_user_controlled(x)

def no_issue_propagation_of_sanitize_all_parameters_no_user_controlled():
    if False:
        while True:
            i = 10
    x = _user_controlled()
    propagation_of_sanitize_all_parameters_no_user_controlled(x)

def issue_propagation_of_sanitize_all_parameters_no_user_controlled():
    if False:
        return 10
    x = _cookies()
    propagation_of_sanitize_all_parameters_no_user_controlled(x)

def sanitize_all_parameters_no_sql(x):
    if False:
        for i in range(10):
            print('nop')
    if 1 > 2:
        _sql(x)
    elif 2 > 3:
        _rce(x)
    else:
        return x

def sanitize_all_parameters_no_rce(x):
    if False:
        return 10
    if 1 > 2:
        _sql(x)
    elif 2 > 3:
        _rce(x)
    else:
        return x

def sanitize_all_parameters_no_user_controlled_tito(x):
    if False:
        while True:
            i = 10
    return x

def no_propagation_with_sanitize_all_parameters_no_user_controlled_tito():
    if False:
        print('Hello World!')
    a = _user_controlled()
    b = sanitize_all_parameters_no_user_controlled_tito(a)
    return b

def propagation_of_cookies_with_sanitize_all_parameters_no_user_controlled_tito():
    if False:
        print('Hello World!')
    b = _cookies()
    tito = sanitize_all_parameters_no_user_controlled_tito(b)
    return tito

def propagation_of_sanitize_user_controlled_tito_in_sink_trace(x):
    if False:
        i = 10
        return i + 15
    y = sanitize_all_parameters_no_user_controlled_tito(x)
    _sql(y)

def sanitize_all_parameters_no_sql_tito(x):
    if False:
        i = 10
        return i + 15
    return x

def no_propagation_with_sanitize_all_parameters_no_sql_tito(x):
    if False:
        while True:
            i = 10
    y = sanitize_all_parameters_no_sql_tito(x)
    _sql(y)

def propagation_of_rce_with_sanitize_all_parameters_no_sql_tito(x):
    if False:
        for i in range(10):
            print('nop')
    y = sanitize_all_parameters_no_sql_tito(x)
    _rce(y)

def propagation_of_sanitize_sql_tito_in_source_trace():
    if False:
        for i in range(10):
            print('nop')
    x = _user_controlled()
    y = sanitize_all_parameters_no_sql_tito(x)
    return y

def no_issue_propagation_of_sanitize_sql_tito_in_source_trace():
    if False:
        print('Hello World!')
    x = propagation_of_sanitize_sql_tito_in_source_trace()
    _sql(x)

def sanitize_all_parameters_no_cookies_sql_tito(x):
    if False:
        for i in range(10):
            print('nop')
    return x

def no_propagation_of_cookies_with_sanitize_all_parameters_no_cookies_sql_tito():
    if False:
        while True:
            i = 10
    a = _cookies()
    b = sanitize_all_parameters_no_cookies_sql_tito(a)
    return b

def propagation_of_user_controlled_with_sanitize_all_parameters_no_cookies_sql_tito():
    if False:
        i = 10
        return i + 15
    b = _user_controlled()
    tito = sanitize_all_parameters_no_cookies_sql_tito(b)
    return tito

def no_propagation_of_sql_with_sanitize_all_parameters_no_cookies_sql_tito(x):
    if False:
        i = 10
        return i + 15
    y = sanitize_all_parameters_no_cookies_sql_tito(x)
    _sql(y)

def propagation_of_rce_with_sanitize_all_parameters_no_cookies_sql_tito(x):
    if False:
        print('Hello World!')
    y = sanitize_all_parameters_no_cookies_sql_tito(x)
    _rce(y)

def sanitize_all_parameters_with_user_declared_sink(x):
    if False:
        while True:
            i = 10
    return x

def sink_taint_sanitize_a(arg):
    if False:
        return 10
    arg = sanitize_a_source_tito(arg)
    _rce(arg)

def sink_taint_sanitize_a_sanitize_b(arg):
    if False:
        return 10
    arg = sanitize_b_source_tito(arg)
    sink_taint_sanitize_a(arg)

def sink_taint_sanitize_a_sanitize_b_santize_test(arg):
    if False:
        while True:
            i = 10
    arg = sanitize_test_source_tito(arg)
    sink_taint_sanitize_a_sanitize_b(arg)

def sink_taint_sanitize_b(arg):
    if False:
        while True:
            i = 10
    arg = sanitize_b_source_tito(arg)
    _rce(arg)

def sink_taint_sanitize_b_sanitize_a(arg):
    if False:
        i = 10
        return i + 15
    arg = sanitize_a_source_tito(arg)
    sink_taint_sanitize_b(arg)

def sink_taint_sanitize_b_sanitize_a_santize_test(arg):
    if False:
        return 10
    arg = sanitize_test_source_tito(arg)
    sink_taint_sanitize_b_sanitize_a(arg)

def sanitize_single_argument_tito(x, y):
    if False:
        for i in range(10):
            print('nop')
    if 1 > 1:
        return x
    else:
        return y