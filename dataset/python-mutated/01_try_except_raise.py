def pop(self):
    if False:
        return 10
    it = iter(self)
    try:
        value = next(it)
    except StopIteration:
        raise KeyError
    self.discard(value)
    return value