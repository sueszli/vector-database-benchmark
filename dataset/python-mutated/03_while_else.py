def acquire(self):
    if False:
        i = 10
        return i + 15
    with self._cond:
        while self:
            rc = False
        else:
            rc = True
    return rc