"""Contains the ObjectPool utility class."""
__all__ = ['Diff', 'ObjectPool']
from direct.directnotify.DirectNotifyGlobal import directNotify
from direct.showbase.PythonUtil import invertDictLossless, makeList, safeRepr, itype
from direct.showbase.PythonUtil import getNumberedTypedString, getNumberedTypedSortedString
import gc

class Diff:

    def __init__(self, lost, gained):
        if False:
            i = 10
            return i + 15
        self.lost = lost
        self.gained = gained

    def printOut(self, full=False):
        if False:
            while True:
                i = 10
        print('lost %s objects, gained %s objects' % (len(self.lost), len(self.gained)))
        print('\n\nself.lost\n')
        print(self.lost.typeFreqStr())
        print('\n\nself.gained\n')
        print(self.gained.typeFreqStr())
        if full:
            self.gained.printObjsByType()
            print('\n\nGAINED-OBJECT REFERRERS\n')
            self.gained.printReferrers(1)

class ObjectPool:
    """manipulate a pool of Python objects"""
    notify = directNotify.newCategory('ObjectPool')

    def __init__(self, objects):
        if False:
            i = 10
            return i + 15
        self._objs = list(objects)
        self._type2objs = {}
        self._count2types = {}
        self._len2obj = {}
        type2count = {}
        for obj in self._objs:
            typ = itype(obj)
            type2count.setdefault(typ, 0)
            type2count[typ] += 1
            self._type2objs.setdefault(typ, [])
            self._type2objs[typ].append(obj)
            try:
                self._len2obj[len(obj)] = obj
            except Exception:
                pass
        self._count2types = invertDictLossless(type2count)

    def _getInternalObjs(self):
        if False:
            while True:
                i = 10
        return (self._objs, self._type2objs, self._count2types)

    def destroy(self):
        if False:
            i = 10
            return i + 15
        del self._objs
        del self._type2objs
        del self._count2types

    def getTypes(self):
        if False:
            print('Hello World!')
        return list(self._type2objs.keys())

    def getObjsOfType(self, type):
        if False:
            i = 10
            return i + 15
        return self._type2objs.get(type, [])

    def printObjsOfType(self, type):
        if False:
            for i in range(10):
                print('nop')
        for obj in self._type2objs.get(type, []):
            print(repr(obj))

    def diff(self, other):
        if False:
            i = 10
            return i + 15
        "print difference between this pool and 'other' pool"
        thisId2obj = {}
        otherId2obj = {}
        for obj in self._objs:
            thisId2obj[id(obj)] = obj
        for obj in other._objs:
            otherId2obj[id(obj)] = obj
        thisIds = set(thisId2obj.keys())
        otherIds = set(otherId2obj.keys())
        lostIds = thisIds.difference(otherIds)
        gainedIds = otherIds.difference(thisIds)
        del thisIds
        del otherIds
        lostObjs = []
        for i in lostIds:
            lostObjs.append(thisId2obj[i])
        gainedObjs = []
        for i in gainedIds:
            gainedObjs.append(otherId2obj[i])
        return Diff(self.__class__(lostObjs), self.__class__(gainedObjs))

    def typeFreqStr(self):
        if False:
            print('Hello World!')
        s = 'Object Pool: Type Frequencies'
        s += '\n============================='
        for count in sorted(self._count2types, reverse=True):
            types = makeList(self._count2types[count])
            for typ in types:
                s += '\n%s\t%s' % (count, typ)
        return s

    def printObjsByType(self):
        if False:
            print('Hello World!')
        print('Object Pool: Objects By Type')
        print('\n============================')
        for count in sorted(self._count2types):
            types = makeList(self._count2types[count])
            for typ in types:
                print('TYPE: %s, %s objects' % (repr(typ), len(self._type2objs[typ])))
                print(getNumberedTypedSortedString(self._type2objs[typ]))

    def printReferrers(self, numEach=3):
        if False:
            for i in range(10):
                print('nop')
        'referrers of the first few of each type of object'
        for count in sorted(self._count2types, reverse=True):
            types = makeList(self._count2types[count])
            for typ in types:
                print('\n\nTYPE: %s' % repr(typ))
                for i in range(min(numEach, len(self._type2objs[typ]))):
                    obj = self._type2objs[typ][i]
                    print('\nOBJ: %s\n' % safeRepr(obj))
                    referrers = gc.get_referrers(obj)
                    print('%s REFERRERS:\n' % len(referrers))
                    if len(referrers) > 0:
                        print(getNumberedTypedString(referrers, maxLen=80, numPrefix='REF'))
                    else:
                        print('<No Referrers>')

    def __len__(self):
        if False:
            print('Hello World!')
        return len(self._objs)