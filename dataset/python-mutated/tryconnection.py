remote = False

def try_connection(verbose, *args, **kwargs):
    if False:
        while True:
            i = 10
    import adodbapi
    dbconnect = adodbapi.connect
    try:
        s = dbconnect(*args, **kwargs)
        if verbose:
            print('Connected to:', s.connection_string)
            print('which has tables:', s.get_table_names())
        s.close()
    except adodbapi.DatabaseError as inst:
        print(inst.args[0])
        print('***Failed getting connection using=', repr(args), repr(kwargs))
        return (False, (args, kwargs), None)
    print('  (successful)')
    return (True, (args, kwargs, remote), dbconnect)

def try_operation_with_expected_exception(expected_exception_list, some_function, *args, **kwargs):
    if False:
        print('Hello World!')
    try:
        some_function(*args, **kwargs)
    except expected_exception_list as e:
        return (True, e)
    except:
        raise
    return (False, 'The expected exception did not occur')