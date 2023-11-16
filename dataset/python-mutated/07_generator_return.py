def __iter__(self):
    if False:
        i = 10
        return i + 15
    i = 0
    try:
        while True:
            v = self[i]
            yield v
            i += 1
    except IndexError:
        return

def iteritems(self):
    if False:
        return 10
    if not self.db:
        return
    try:
        try:
            yield self.kv
        except:
            pass
    except:
        self._in_iter -= 1
        raise