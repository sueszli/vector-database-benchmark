def capture_error(callable):
    if False:
        i = 10
        return i + 15
    try:
        callable()
    except Exception as e:
        return e
    return None