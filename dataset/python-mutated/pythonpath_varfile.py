def get_variables(*args):
    if False:
        for i in range(10):
            print('nop')
    return {'PYTHONPATH VAR %d' % len(args): 'Varfile found from PYTHONPATH', 'PYTHONPATH ARGS %d' % len(args): '-'.join(args)}