class TreeNode(object):

    def __init__(self, x):
        if False:
            return 10
        self.val = x
        self.left = None
        self.right = None

def serialize(root):
    if False:
        i = 10
        return i + 15

    def build_string(node):
        if False:
            while True:
                i = 10
        if node:
            vals.append(str(node.val))
            build_string(node.left)
            build_string(node.right)
        else:
            vals.append('#')
    vals = []
    build_string(root)
    return ' '.join(vals)

def deserialize(data):
    if False:
        while True:
            i = 10

    def build_tree():
        if False:
            for i in range(10):
                print('nop')
        val = next(vals)
        if val == '#':
            return None
        node = TreeNode(int(val))
        node.left = build_tree()
        node.right = build_tree()
        return node
    vals = iter(data.split())
    return build_tree()