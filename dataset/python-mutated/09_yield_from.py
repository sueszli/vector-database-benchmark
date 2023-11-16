def _walk_dir(dir, dfile, ddir=None):
    if False:
        for i in range(10):
            print('nop')
    yield from _walk_dir(dir, ddir=dfile)

def ybug(g):
    if False:
        print('Hello World!')
    yield from g

def __iter__(self, IterationGuard):
    if False:
        return 10
    with IterationGuard(self):
        for itemref in self.data:
            item = itemref()
            if item is not None:
                yield item