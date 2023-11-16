class Node(object):

    def __init__(self, val, children):
        if False:
            print('Hello World!')
        self.val = val
        self.children = children

class Codec(object):

    def serialize(self, root):
        if False:
            return 10
        'Encodes a tree to a single string.\n        \n        :type root: Node\n        :rtype: str\n        '

        def dfs(node, vals):
            if False:
                for i in range(10):
                    print('nop')
            if not node:
                return
            vals.append(str(node.val))
            for child in node.children:
                dfs(child, vals)
            vals.append('#')
        vals = []
        dfs(root, vals)
        return ' '.join(vals)

    def deserialize(self, data):
        if False:
            return 10
        'Decodes your encoded data to tree.\n        \n        :type data: str\n        :rtype: Node\n        '

        def isplit(source, sep):
            if False:
                i = 10
                return i + 15
            sepsize = len(sep)
            start = 0
            while True:
                idx = source.find(sep, start)
                if idx == -1:
                    yield source[start:]
                    return
                yield source[start:idx]
                start = idx + sepsize

        def dfs(vals):
            if False:
                for i in range(10):
                    print('nop')
            val = next(vals)
            if val == '#':
                return None
            root = Node(int(val), [])
            child = dfs(vals)
            while child:
                root.children.append(child)
                child = dfs(vals)
            return root
        if not data:
            return None
        return dfs(iter(isplit(data, ' ')))