class TreeNode(object):

    def __init__(self, x):
        if False:
            i = 10
            return i + 15
        self.val = x
        self.left = None
        self.right = None

class Codec(object):

    def serialize(self, root):
        if False:
            for i in range(10):
                print('nop')
        'Encodes a tree to a single string.\n\n        :type root: TreeNode\n        :rtype: str\n        '

        def serializeHelper(node):
            if False:
                return 10
            if not node:
                vals.append('#')
                return
            vals.append(str(node.val))
            serializeHelper(node.left)
            serializeHelper(node.right)
        vals = []
        serializeHelper(root)
        return ' '.join(vals)

    def deserialize(self, data):
        if False:
            i = 10
            return i + 15
        'Decodes your encoded data to tree.\n\n        :type data: str\n        :rtype: TreeNode\n        '

        def deserializeHelper():
            if False:
                return 10
            val = next(vals)
            if val == '#':
                return None
            node = TreeNode(int(val))
            node.left = deserializeHelper()
            node.right = deserializeHelper()
            return node

        def isplit(source, sep):
            if False:
                for i in range(10):
                    print('nop')
            sepsize = len(sep)
            start = 0
            while True:
                idx = source.find(sep, start)
                if idx == -1:
                    yield source[start:]
                    return
                yield source[start:idx]
                start = idx + sepsize
        vals = iter(isplit(data, ' '))
        return deserializeHelper()

class Codec2(object):

    def serialize(self, root):
        if False:
            while True:
                i = 10
        'Encodes a tree to a single string.\n        \n        :type root: TreeNode\n        :rtype: str\n        '

        def gen_preorder(node):
            if False:
                for i in range(10):
                    print('nop')
            if not node:
                yield '#'
            else:
                yield str(node.val)
                for n in gen_preorder(node.left):
                    yield n
                for n in gen_preorder(node.right):
                    yield n
        return ' '.join(gen_preorder(root))

    def deserialize(self, data):
        if False:
            for i in range(10):
                print('nop')
        'Decodes your encoded data to tree.\n        \n        :type data: str\n        :rtype: TreeNode\n        '

        def builder(chunk_iter):
            if False:
                while True:
                    i = 10
            val = next(chunk_iter)
            if val == '#':
                return None
            node = TreeNode(int(val))
            node.left = builder(chunk_iter)
            node.right = builder(chunk_iter)
            return node
        chunk_iter = iter(data.split())
        return builder(chunk_iter)