import sys
from threading import Lock
import hashlib
from xml.etree import cElementTree as ET
from .exception import TarsException

class util:

    @staticmethod
    def printHex(buff):
        if False:
            print('Hello World!')
        count = 0
        for c in buff:
            sys.stdout.write('0X%02X ' % ord(c))
            count += 1
            if count % 16 == 0:
                sys.stdout.write('\n')
        sys.stdout.write('\n')
        sys.stdout.flush()

    @staticmethod
    def mapclass(ktype, vtype):
        if False:
            return 10

        class mapklass(dict):

            def size(self):
                if False:
                    for i in range(10):
                        print('nop')
                return len(self)
        setattr(mapklass, '__tars_index__', 8)
        setattr(mapklass, '__tars_class__', 'map<' + ktype.__tars_class__ + ',' + vtype.__tars_class__ + '>')
        setattr(mapklass, 'ktype', ktype)
        setattr(mapklass, 'vtype', vtype)
        return mapklass

    @staticmethod
    def vectorclass(vtype):
        if False:
            while True:
                i = 10

        class klass(list):

            def size(self):
                if False:
                    return 10
                return len(self)
        setattr(klass, '__tars_index__', 9)
        setattr(klass, '__tars_class__', 'list<' + vtype.__tars_class__ + '>')
        setattr(klass, 'vtype', vtype)
        return klass

    class boolean:
        __tars_index__ = 999
        __tars_class__ = 'bool'

    class int8:
        __tars_index__ = 0
        __tars_class__ = 'char'

    class uint8:
        __tars_index__ = 1
        __tars_class__ = 'short'

    class int16:
        __tars_index__ = 1
        __tars_class__ = 'short'

    class uint16:
        __tars_index__ = 2
        __tars_class__ = 'int32'

    class int32:
        __tars_index__ = 2
        __tars_class__ = 'int32'

    class uint32:
        __tars_index__ = 3
        __tars_class__ = 'int64'

    class int64:
        __tars_index__ = 3
        __tars_class__ = 'int64'

    class float:
        __tars_index__ = 4
        __tars_class__ = 'float'

    class double:
        __tars_index__ = 5
        __tars_class__ = 'double'

    class bytes:
        __tars_index__ = 13
        __tars_class__ = 'list<char>'

    class string:
        __tars_index__ = 67
        __tars_class__ = 'string'

    class struct:
        __tars_index__ = 1011

def xml2dict(node, dic={}):
    if False:
        return 10
    '\n    @brief: 将xml解析树转成字典\n    @param node: 树的根节点\n    @type node: cElementTree.Element\n    @param dic: 存储信息的字典\n    @type dic: dict\n    @return: 转换好的字典\n    @rtype: dict\n    '
    dic[node.tag] = ndic = {}
    [xml2dict(child, ndic) for child in node.getchildren() if child != node]
    ndic.update([list(map(str.strip, exp.split('=')[:2])) for exp in node.text.splitlines() if '=' in exp])
    return dic

def configParse(filename):
    if False:
        i = 10
        return i + 15
    '\n    @brief: 解析tars配置文件\n    @param filename: 文件名\n    @type filename: str\n    @return: 解析出来的配置信息\n    @rtype: dict\n    '
    tree = ET.parse(filename)
    return xml2dict(tree.getroot())

class NewLock(object):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.__count = 0
        self.__lock = Lock()
        self.__lockForCount = Lock()
        pass

    def newAcquire(self):
        if False:
            print('Hello World!')
        self.__lockForCount.acquire()
        self.__count += 1
        if self.__count == 1:
            self.__lock.acquire()
        self.__lockForCount.release()
        pass

    def newRelease(self):
        if False:
            return 10
        self.__lockForCount.acquire()
        self.__count -= 1
        if self.__count == 0:
            self.__lock.release()
        self.__lockForCount.release()

class LockGuard(object):

    def __init__(self, newLock):
        if False:
            print('Hello World!')
        self.__newLock = newLock
        self.__newLock.newAcquire()

    def __del__(self):
        if False:
            while True:
                i = 10
        self.__newLock.newRelease()

class ConsistentHashNew(object):

    def __init__(self, nodes=None, nodeNumber=3):
        if False:
            print('Hello World!')
        '\n        :param nodes:           服务器的节点的epstr列表\n        :param n_number:        一个节点对应的虚拟节点数量\n        :return:\n        '
        self.__nodes = nodes
        self.__nodeNumber = nodeNumber
        self.__nodeDict = dict()
        self.__sortListForKey = []
        if nodes:
            for node in nodes:
                self.addNode(node)

    @property
    def nodes(self):
        if False:
            print('Hello World!')
        return self.__nodes

    @nodes.setter
    def nodes(self, value):
        if False:
            print('Hello World!')
        self.__nodes = value

    def addNode(self, node):
        if False:
            return 10
        '\n        添加node，首先要根据虚拟节点的数目，创建所有的虚拟节点，并将其与对应的node对应起来\n        当然还需要将虚拟节点的hash值放到排序的里面\n        这里在添加了节点之后，需要保持虚拟节点hash值的顺序\n        :param node:\n        :return:\n        '
        for i in range(self.__nodeNumber):
            nodeStr = '%s%s' % (node, i)
            key = self.__genKey(nodeStr)
            self.__nodeDict[key] = node
            self.__sortListForKey.append(key)
        self.__sortListForKey.sort()

    def removeNode(self, node):
        if False:
            return 10
        '\n        这里一个节点的退出，需要将这个节点的所有的虚拟节点都删除\n        :param node:\n        :return:\n        '
        for i in range(self.__nodeNumber):
            nodeStr = '%s%s' % (node, i)
            key = self.__genKey(nodeStr)
            del self.__nodeDict[key]
            self.__sortListForKey.remove(key)

    def getNode(self, key):
        if False:
            i = 10
            return i + 15
        '\n        返回这个字符串应该对应的node，这里先求出字符串的hash值，然后找到第一个小于等于的虚拟节点，然后返回node\n        如果hash值大于所有的节点，那么用第一个虚拟节点\n        :param : hashNum or keyStr\n        :return:\n        '
        keyStr = ''
        if isinstance(key, int):
            keyStr = 'the keyStr is %d' % key
        elif isinstance(key, type('a')):
            keyStr = key
        else:
            raise TarsException('the hash code has wrong type')
        if self.__sortListForKey:
            key = self.__genKey(keyStr)
            for keyItem in self.__sortListForKey:
                if key <= keyItem:
                    return self.__nodeDict[keyItem]
            return self.__nodeDict[self.__sortListForKey[0]]
        else:
            return None

    def __genKey(self, keyStr):
        if False:
            return 10
        '\n        通过key，返回当前key的hash值，这里采用md5\n        :param key:\n        :return:\n        '
        md5Str = hashlib.md5(keyStr).hexdigest()
        return int(md5Str, 16)