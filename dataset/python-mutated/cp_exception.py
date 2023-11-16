def throw_must_not_go_through_else():
    if False:
        for i in range(10):
            print('nop')
    x = 0
    try:
        raise RuntimeError()
    except Exception as e:
        x = 0
    else:
        x = 9999
    x == 0

def no_throw_goes_through_else():
    if False:
        print('Hello World!')
    x = 0
    try:
        pass
    except Exception as e:
        x = 9999
    else:
        x = 0
    x == 0

def may_throw_goes_through_catch_and_else():
    if False:
        print('Hello World!')
    x = 0
    try:
        any_function_call_may_raise()
    except Exception as e:
        x = 0
    else:
        x = 1
    x == 0

def exception_or_not_goes_through_finally():
    if False:
        while True:
            i = 10
    x = 0
    try:
        any_function_call_may_raise()
    except Exception as e:
        x = 0
    else:
        x = 0
    finally:
        x = 1
    x == 0

def non_nested_try_statements_are_independent():
    if False:
        return 10
    x = 0
    try:
        try:
            any_function_call_may_raise()
        finally:
            pass
    except Exception as e:
        x = 1
    x == 0
    y = 0
    try:
        try:
            pass
        finally:
            pass
    except Exception as e:
        y = 1
    y == 0