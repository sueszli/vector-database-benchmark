def a():
    if False:
        i = 10
        return i + 15
    try:
        bad
    except Exception:
        logger.error('Something bad happened')