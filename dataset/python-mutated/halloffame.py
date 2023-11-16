from operator import attrgetter
from conf import runid

class ObjectiveFunctionHallOfFame(object):

    def __init__(self, maxsize=30):
        if False:
            for i in range(10):
                print('nop')
        self.inner = set()
        self.maxsize = maxsize

    def update(self, newpop):
        if False:
            while True:
                i = 10
        self.inner = self.inner.union(newpop)
        self.inner = set(sorted(self.inner, key=attrgetter('objective'), reverse=True)[:self.maxsize])

    def __iter__(self):
        if False:
            print('Hello World!')
        return iter(self.inner)

    def len(self):
        if False:
            return 10
        return len(self.inner)

    def __repr__(self):
        if False:
            return 10
        header = ['Current Hall of Fame:']
        report = [f'{ind}' for ind in sorted(self.inner, key=attrgetter('objective'), reverse=True)]
        return '\n'.join(header + report)

    def persist(self):
        if False:
            print('Hello World!')
        with open('logs/hof/{runid}.txt'.format(runid=runid), 'w') as f:
            f.write(str(self))