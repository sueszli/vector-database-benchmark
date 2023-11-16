class TrieNode(object):

    def __init__(self):
        if False:
            return 10
        self.is_file = False
        self.children = {}
        self.content = ''

class FileSystem(object):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.__root = TrieNode()

    def ls(self, path):
        if False:
            print('Hello World!')
        '\n        :type path: str\n        :rtype: List[str]\n        '
        curr = self.__getNode(path)
        if curr.is_file:
            return [self.__split(path, '/')[-1]]
        return sorted(curr.children.keys())

    def mkdir(self, path):
        if False:
            print('Hello World!')
        '\n        :type path: str\n        :rtype: void\n        '
        curr = self.__putNode(path)
        curr.is_file = False

    def addContentToFile(self, filePath, content):
        if False:
            return 10
        '\n        :type filePath: str\n        :type content: str\n        :rtype: void\n        '
        curr = self.__putNode(filePath)
        curr.is_file = True
        curr.content += content

    def readContentFromFile(self, filePath):
        if False:
            i = 10
            return i + 15
        '\n        :type filePath: str\n        :rtype: str\n        '
        return self.__getNode(filePath).content

    def __getNode(self, path):
        if False:
            while True:
                i = 10
        curr = self.__root
        for s in self.__split(path, '/'):
            curr = curr.children[s]
        return curr

    def __putNode(self, path):
        if False:
            i = 10
            return i + 15
        curr = self.__root
        for s in self.__split(path, '/'):
            if s not in curr.children:
                curr.children[s] = TrieNode()
            curr = curr.children[s]
        return curr

    def __split(self, path, delim):
        if False:
            for i in range(10):
                print('nop')
        if path == '/':
            return []
        return path.split('/')[1:]