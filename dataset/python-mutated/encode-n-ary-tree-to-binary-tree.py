class Node(object):

    def __init__(self, val, children):
        if False:
            return 10
        self.val = val
        self.children = children

class TreeNode(object):

    def __init__(self, x):
        if False:
            i = 10
            return i + 15
        self.val = x
        self.left = None
        self.right = None

class Codec(object):

    def encode(self, root):
        if False:
            return 10
        'Encodes an n-ary tree to a binary tree.\n        \n        :type root: Node\n        :rtype: TreeNode\n        '

        def encodeHelper(root, parent, index):
            if False:
                for i in range(10):
                    print('nop')
            if not root:
                return None
            node = TreeNode(root.val)
            if index + 1 < len(parent.children):
                node.left = encodeHelper(parent.children[index + 1], parent, index + 1)
            if root.children:
                node.right = encodeHelper(root.children[0], root, 0)
            return node
        if not root:
            return None
        node = TreeNode(root.val)
        if root.children:
            node.right = encodeHelper(root.children[0], root, 0)
        return node

    def decode(self, data):
        if False:
            i = 10
            return i + 15
        'Decodes your binary tree to an n-ary tree.\n        \n        :type data: TreeNode\n        :rtype: Node\n        '

        def decodeHelper(root, parent):
            if False:
                for i in range(10):
                    print('nop')
            if not root:
                return
            children = []
            node = Node(root.val, children)
            decodeHelper(root.right, node)
            parent.children.append(node)
            decodeHelper(root.left, parent)
        if not data:
            return None
        children = []
        node = Node(data.val, children)
        decodeHelper(data.right, node)
        return node