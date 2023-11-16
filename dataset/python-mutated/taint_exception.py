def throw_exits(input):
    if False:
        for i in range(10):
            print('nop')
    clean = None
    raise RuntimeError()
    sink(clean)

def throw_in_catch_exits(input):
    if False:
        print('Hello World!')
    clean = None
    try:
        raise RuntimeError()
    except Exception as e:
        raise RuntimeError()
    sink(clean)

def throw_in_else_exits(input):
    if False:
        i = 10
        return i + 15
    clean = None
    try:
        pass
    except Exception as e:
        clean = input
        sink(clean)
    else:
        raise RuntimeError()
    sink(clean)

def throw_in_finally_exits(input):
    if False:
        for i in range(10):
            print('nop')
    clean = None
    try:
        pass
    finally:
        raise RuntimeError()
    sink(clean)

def return_exits(input):
    if False:
        print('Hello World!')
    clean = None
    return
    sink(clean)

def return_in_catch_exits(input):
    if False:
        i = 10
        return i + 15
    clean = None
    try:
        raise RuntimeError()
    except Exception as e:
        return
    sink(clean)

def return_in_else_exits(input):
    if False:
        i = 10
        return i + 15
    clean = None
    try:
        pass
    except Exception as e:
        clean = input
        sink(clean)
    else:
        return
    sink(clean)

def throw_in_finally_exits(input):
    if False:
        i = 10
        return i + 15
    clean = None
    try:
        pass
    finally:
        return
    sink(clean)

def throw_must_not_go_through_else(input):
    if False:
        while True:
            i = 10
    clean = None
    dirty = None
    try:
        raise RuntimeError()
    except Exception as e:
        dirty = input
    else:
        clean = input
    sink(clean)
    sink(dirty)

def no_throw_goes_through_else(input):
    if False:
        print('Hello World!')
    clean = None
    dirty = None
    try:
        pass
    except Exception as e:
        clean = input
    else:
        dirty = input
    sink(clean)
    sink(dirty)

def may_throw_goes_through_catch_and_else(input):
    if False:
        return 10
    dirty1 = None
    dirty2 = None
    try:
        any_function_call_may_raise()
    except Exception as e:
        dirty1 = input
    else:
        dirty2 = input
    sink(dirty1)
    sink(dirty2)

def exception_or_not_goes_through_finally(input):
    if False:
        while True:
            i = 10
    clean1 = None
    clean2 = None
    try:
        any_function_call_may_raise()
    except Exception as e:
        clean1 = input
    else:
        clean2 = input
    finally:
        clean1 = sanitize(clean1)
        clean2 = sanitize(clean2)
    sink(clean1)
    sink(clean2)

def throw_may_go_through_catch_and_propagates(input):
    if False:
        return 10
    clean1 = None
    clean2 = None
    clean3 = None
    dirty1 = None
    dirty2 = None
    dirty3 = None
    try:
        try:
            raise RuntimeError()
        except Exception as e:
            clean1 = input
            sink(clean1)
        else:
            clean2 = input
            sink(clean2)
        finally:
            dirty1 = input
        clean3 = input
    except Exception as e:
        dirty2 = input
    finally:
        dirty3 = input
        sink(dirty1)
        sink(dirty2)
        sink(dirty3)