"""Functions that involve magic. """

def pollute(names, objects):
    if False:
        for i in range(10):
            print('nop')
    'Pollute the global namespace with symbols -> objects mapping. '
    from inspect import currentframe
    frame = currentframe().f_back.f_back
    try:
        for (name, obj) in zip(names, objects):
            frame.f_globals[name] = obj
    finally:
        del frame