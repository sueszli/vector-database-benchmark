def matchTuple(routingKey, filter):
    if False:
        return 10
    if len(filter) != len(routingKey):
        return False
    for (k, f) in zip(routingKey, filter):
        if f is not None and f != k:
            return False
    return True