import inspect
import sys
import gc
from direct.showbase.PythonUtil import _getSafeReprNotify, fastRepr
from direct.showbase.Job import Job
from direct.showbase.MessengerGlobal import messenger
from direct.task.TaskManagerGlobal import taskMgr

class ReferrerSearch(Job):

    def __init__(self, obj, maxRefs=100):
        if False:
            while True:
                i = 10
        Job.__init__(self, 'ReferrerSearch')
        self.obj = obj
        self.maxRefs = maxRefs
        self.visited = set()
        self.depth = 0
        self.found = 0
        self.shouldPrintStats = False

    def __call__(self):
        if False:
            return 10
        safeReprNotify = _getSafeReprNotify()
        info = safeReprNotify.getInfo()
        safeReprNotify.setInfo(0)
        self.visited = set()
        try:
            self.step(0, [self.obj])
        finally:
            self.obj = None
        safeReprNotify.setInfo(info)

    def run(self):
        if False:
            return 10
        safeReprNotify = _getSafeReprNotify()
        self.info = safeReprNotify.getInfo()
        safeReprNotify.setInfo(0)
        print('RefPath(%s): Beginning ReferrerSearch for %s' % (self._id, fastRepr(self.obj)))
        self.visited = set()
        for x in self.stepGenerator(0, [self.obj]):
            yield None
        yield Job.Done

    def finished(self):
        if False:
            print('Hello World!')
        print('RefPath(%s): Finished ReferrerSearch for %s' % (self._id, fastRepr(self.obj)))
        self.obj = None
        safeReprNotify = _getSafeReprNotify()
        safeReprNotify.setInfo(self.info)

    def __del__(self):
        if False:
            while True:
                i = 10
        print('ReferrerSearch garbage collected')

    def truncateAtNewLine(self, s):
        if False:
            i = 10
            return i + 15
        if s.find('\n') == -1:
            return s
        else:
            return s[:s.find('\n')]

    def printStatsWhenAble(self):
        if False:
            i = 10
            return i + 15
        self.shouldPrintStats = True

    def myrepr(self, referrer, refersTo):
        if False:
            print('Hello World!')
        pre = ''
        if isinstance(referrer, dict):
            for (k, v) in referrer.items():
                if v is refersTo:
                    pre = self.truncateAtNewLine(fastRepr(k)) + ']-> '
                    break
        elif isinstance(referrer, (list, tuple)):
            for (x, ref) in enumerate(referrer):
                if ref is refersTo:
                    pre = '%s]-> ' % x
                    break
        if isinstance(refersTo, dict):
            post = 'dict['
        elif isinstance(refersTo, list):
            post = 'list['
        elif isinstance(refersTo, tuple):
            post = 'tuple['
        elif isinstance(refersTo, set):
            post = 'set->'
        else:
            post = self.truncateAtNewLine(fastRepr(refersTo)) + '-> '
        return '%s%s' % (pre, post)

    def step(self, depth, path):
        if False:
            for i in range(10):
                print('nop')
        if self.shouldPrintStats:
            self.printStats(path)
            self.shouldPrintStats = False
        at = path[-1]
        if id(at) in self.visited:
            return
        if self.isAtRoot(at, path):
            self.found += 1
            return
        self.visited.add(id(at))
        referrers = [ref for ref in gc.get_referrers(at) if not (ref is path or inspect.isframe(ref) or (isinstance(ref, dict) and list(ref.keys()) == list(locals().keys())) or (ref is self.__dict__) or (id(ref) in self.visited))]
        if self.isManyRef(at, path, referrers):
            return
        while referrers:
            ref = referrers.pop()
            self.depth += 1
            for x in self.stepGenerator(depth + 1, path + [ref]):
                pass
            self.depth -= 1

    def stepGenerator(self, depth, path):
        if False:
            i = 10
            return i + 15
        if self.shouldPrintStats:
            self.printStats(path)
            self.shouldPrintStats = False
        at = path[-1]
        if self.isAtRoot(at, path):
            self.found += 1
            raise StopIteration
        if id(at) in self.visited:
            raise StopIteration
        self.visited.add(id(at))
        referrers = [ref for ref in gc.get_referrers(at) if not (ref is path or inspect.isframe(ref) or (isinstance(ref, dict) and list(ref.keys()) == list(locals().keys())) or (ref is self.__dict__) or (id(ref) in self.visited))]
        if self.isManyRef(at, path, referrers):
            raise StopIteration
        while referrers:
            ref = referrers.pop()
            self.depth += 1
            for x in self.stepGenerator(depth + 1, path + [ref]):
                yield None
            self.depth -= 1
        yield None

    def printStats(self, path):
        if False:
            i = 10
            return i + 15
        path = list(reversed(path))
        path.insert(0, 0)
        print('RefPath(%s) - Stats - visited(%s) | found(%s) | depth(%s) | CurrentPath(%s)' % (self._id, len(self.visited), self.found, self.depth, ''.join((self.myrepr(path[x], path[x + 1]) for x in range(len(path) - 1)))))

    def isAtRoot(self, at, path):
        if False:
            for i in range(10):
                print('nop')
        if at in path:
            sys.stdout.write('RefPath(%s): Circular: ' % self._id)
            path = list(reversed(path))
            path.insert(0, 0)
            for x in range(len(path) - 1):
                sys.stdout.write(self.myrepr(path[x], path[x + 1]))
            print('')
            return True
        if at is __builtins__:
            sys.stdout.write('RefPath(%s): __builtins__-> ' % self._id)
            path = list(reversed(path))
            path.insert(0, 0)
            for x in range(len(path) - 1):
                sys.stdout.write(self.myrepr(path[x], path[x + 1]))
            print('')
            return True
        if inspect.ismodule(at):
            sys.stdout.write('RefPath(%s): Module(%s)-> ' % (self._id, at.__name__))
            path = list(reversed(path))
            for x in range(len(path) - 1):
                sys.stdout.write(self.myrepr(path[x], path[x + 1]))
            print('')
            return True
        if inspect.isclass(at):
            sys.stdout.write('RefPath(%s): Class(%s)-> ' % (self._id, at.__name__))
            path = list(reversed(path))
            for x in range(len(path) - 1):
                sys.stdout.write(self.myrepr(path[x], path[x + 1]))
            print('')
            return True
        if at is simbase:
            sys.stdout.write('RefPath(%s): simbase-> ' % self._id)
            path = list(reversed(path))
            for x in range(len(path) - 1):
                sys.stdout.write(self.myrepr(path[x], path[x + 1]))
            print('')
            return True
        if at is simbase.air:
            sys.stdout.write('RefPath(%s): simbase.air-> ' % self._id)
            path = list(reversed(path))
            for x in range(len(path) - 1):
                sys.stdout.write(self.myrepr(path[x], path[x + 1]))
            print('')
            return True
        if at is messenger:
            sys.stdout.write('RefPath(%s): messenger-> ' % self._id)
            path = list(reversed(path))
            for x in range(len(path) - 1):
                sys.stdout.write(self.myrepr(path[x], path[x + 1]))
            print('')
            return True
        if at is taskMgr:
            sys.stdout.write('RefPath(%s): taskMgr-> ' % self._id)
            path = list(reversed(path))
            for x in range(len(path) - 1):
                sys.stdout.write(self.myrepr(path[x], path[x + 1]))
            print('')
            return True
        if hasattr(simbase.air, 'mainWorld') and at is simbase.air.mainWorld:
            sys.stdout.write('RefPath(%s): mainWorld-> ' % self._id)
            path = list(reversed(path))
            for x in range(len(path) - 1):
                sys.stdout.write(self.myrepr(path[x], path[x + 1]))
            print('')
            return True
        return False

    def isManyRef(self, at, path, referrers):
        if False:
            return 10
        if len(referrers) > self.maxRefs and at is not self.obj:
            if not isinstance(at, (list, tuple, dict, set)):
                sys.stdout.write('RefPath(%s): ManyRefs(%s)[%s]-> ' % (self._id, len(referrers), fastRepr(at)))
                path = list(reversed(path))
                path.insert(0, 0)
                for x in range(len(path) - 1):
                    sys.stdout.write(self.myrepr(path[x], path[x + 1]))
                print('')
                return True
            else:
                sys.stdout.write('RefPath(%s): ManyRefsAllowed(%s)[%s]-> ' % (self._id, len(referrers), fastRepr(at, maxLen=1, strFactor=30)))
                print('')
        return False