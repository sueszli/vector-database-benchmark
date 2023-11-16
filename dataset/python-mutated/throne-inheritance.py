import collections

class ThroneInheritance(object):

    def __init__(self, kingName):
        if False:
            while True:
                i = 10
        '\n        :type kingName: str\n        '
        self.__king = kingName
        self.__family_tree = collections.defaultdict(list)
        self.__dead = set()

    def birth(self, parentName, childName):
        if False:
            return 10
        '\n        :type parentName: str\n        :type childName: str\n        :rtype: None\n        '
        self.__family_tree[parentName].append(childName)

    def death(self, name):
        if False:
            while True:
                i = 10
        '\n        :type name: str\n        :rtype: None\n        '
        self.__dead.add(name)

    def getInheritanceOrder(self):
        if False:
            return 10
        '\n        :rtype: List[str]\n        '
        result = []
        stk = [self.__king]
        while stk:
            node = stk.pop()
            if node not in self.__dead:
                result.append(node)
            if node not in self.__family_tree:
                continue
            for child in reversed(self.__family_tree[node]):
                stk.append(child)
        return result