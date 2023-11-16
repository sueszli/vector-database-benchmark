def accept():
    if False:
        return 10
    try:
        conn = 5
    except TypeError:
        return None
    except OSError as why:
        if why == 6:
            raise
    else:
        return conn