import collections

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
            print('Hello World!')
        'Encodes a tree to a single string.\n\n        :type root: TreeNode\n        :rtype: str\n        '

        def serializeHelper(node, vals):
            if False:
                i = 10
                return i + 15
            if node:
                vals.append(node.val)
                serializeHelper(node.left, vals)
                serializeHelper(node.right, vals)
        vals = []
        serializeHelper(root, vals)
        return ' '.join(map(str, vals))

    def deserialize(self, data):
        if False:
            i = 10
            return i + 15
        'Decodes your encoded data to tree.\n\n        :type data: str\n        :rtype: TreeNode\n        '

        def deserializeHelper(minVal, maxVal, vals):
            if False:
                i = 10
                return i + 15
            if not vals:
                return None
            if minVal < vals[0] < maxVal:
                val = vals.popleft()
                node = TreeNode(val)
                node.left = deserializeHelper(minVal, val, vals)
                node.right = deserializeHelper(val, maxVal, vals)
                return node
            else:
                return None
        vals = collections.deque([int(val) for val in data.split()])
        return deserializeHelper(float('-inf'), float('inf'), vals)