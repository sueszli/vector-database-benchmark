"""
Implementation of Red-Black tree.
"""

class RBNode:

    def __init__(self, val, is_red, parent=None, left=None, right=None):
        if False:
            for i in range(10):
                print('nop')
        self.val = val
        self.parent = parent
        self.left = left
        self.right = right
        self.color = is_red

class RBTree:

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.root = None

    def left_rotate(self, node):
        if False:
            return 10
        right_node = node.right
        if right_node is None:
            return
        else:
            node.right = right_node.left
            if right_node.left is not None:
                right_node.left.parent = node
            right_node.parent = node.parent
            if node.parent is None:
                self.root = right_node
            elif node is node.parent.left:
                node.parent.left = right_node
            else:
                node.parent.right = right_node
            right_node.left = node
            node.parent = right_node

    def right_rotate(self, node):
        if False:
            return 10
        left_node = node.left
        if left_node is None:
            return
        else:
            node.left = left_node.right
            if left_node.right is not None:
                left_node.right.parent = node
            left_node.parent = node.parent
            if node.parent is None:
                self.root = left_node
            elif node is node.parent.left:
                node.parent.left = left_node
            else:
                node.parent.right = left_node
            left_node.right = node
            node.parent = left_node

    def insert(self, node):
        if False:
            for i in range(10):
                print('nop')
        root = self.root
        insert_node_parent = None
        while root is not None:
            insert_node_parent = root
            if insert_node_parent.val < node.val:
                root = root.right
            else:
                root = root.left
        node.parent = insert_node_parent
        if insert_node_parent is None:
            self.root = node
        elif insert_node_parent.val > node.val:
            insert_node_parent.left = node
        else:
            insert_node_parent.right = node
        node.left = None
        node.right = None
        node.color = 1
        self.fix_insert(node)

    def fix_insert(self, node):
        if False:
            return 10
        if node.parent is None:
            node.color = 0
            self.root = node
            return
        while node.parent and node.parent.color == 1:
            if node.parent is node.parent.parent.left:
                uncle_node = node.parent.parent.right
                if uncle_node and uncle_node.color == 1:
                    node.parent.color = 0
                    node.parent.parent.right.color = 0
                    node.parent.parent.color = 1
                    node = node.parent.parent
                    continue
                elif node is node.parent.right:
                    node = node.parent
                    self.left_rotate(node)
                node.parent.color = 0
                node.parent.parent.color = 1
                self.right_rotate(node.parent.parent)
            else:
                uncle_node = node.parent.parent.left
                if uncle_node and uncle_node.color == 1:
                    node.parent.color = 0
                    node.parent.parent.left.color = 0
                    node.parent.parent.color = 1
                    node = node.parent.parent
                    continue
                elif node is node.parent.left:
                    node = node.parent
                    self.right_rotate(node)
                node.parent.color = 0
                node.parent.parent.color = 1
                self.left_rotate(node.parent.parent)
        self.root.color = 0

    def transplant(self, node_u, node_v):
        if False:
            i = 10
            return i + 15
        '\n        replace u with v\n        :param node_u: replaced node\n        :param node_v: \n        :return: None\n        '
        if node_u.parent is None:
            self.root = node_v
        elif node_u is node_u.parent.left:
            node_u.parent.left = node_v
        elif node_u is node_u.parent.right:
            node_u.parent.right = node_v
        if node_v:
            node_v.parent = node_u.parent

    def maximum(self, node):
        if False:
            for i in range(10):
                print('nop')
        '\n        find the max node when node regard as a root node   \n        :param node: \n        :return: max node \n        '
        temp_node = node
        while temp_node.right is not None:
            temp_node = temp_node.right
        return temp_node

    def minimum(self, node):
        if False:
            for i in range(10):
                print('nop')
        '\n        find the minimum node when node regard as a root node   \n        :param node:\n        :return: minimum node \n        '
        temp_node = node
        while temp_node.left:
            temp_node = temp_node.left
        return temp_node

    def delete(self, node):
        if False:
            print('Hello World!')
        node_color = node.color
        if node.left is None:
            temp_node = node.right
            self.transplant(node, node.right)
        elif node.right is None:
            temp_node = node.left
            self.transplant(node, node.left)
        else:
            node_min = self.minimum(node.right)
            node_color = node_min.color
            temp_node = node_min.right
            if node_min.parent is not node:
                self.transplant(node_min, node_min.right)
                node_min.right = node.right
                node_min.right.parent = node_min
            self.transplant(node, node_min)
            node_min.left = node.left
            node_min.left.parent = node_min
            node_min.color = node.color
        if node_color == 0:
            self.delete_fixup(temp_node)

    def delete_fixup(self, node):
        if False:
            for i in range(10):
                print('nop')
        while node is not self.root and node.color == 0:
            if node is node.parent.left:
                node_brother = node.parent.right
                if node_brother.color == 1:
                    node_brother.color = 0
                    node.parent.color = 1
                    self.left_rotate(node.parent)
                    node_brother = node.parent.right
                if (node_brother.left is None or node_brother.left.color == 0) and (node_brother.right is None or node_brother.right.color == 0):
                    node_brother.color = 1
                    node = node.parent
                else:
                    if node_brother.right is None or node_brother.right.color == 0:
                        node_brother.color = 1
                        node_brother.left.color = 0
                        self.right_rotate(node_brother)
                        node_brother = node.parent.right
                    node_brother.color = node.parent.color
                    node.parent.color = 0
                    node_brother.right.color = 0
                    self.left_rotate(node.parent)
                    node = self.root
            else:
                node_brother = node.parent.left
                if node_brother.color == 1:
                    node_brother.color = 0
                    node.parent.color = 1
                    self.left_rotate(node.parent)
                    node_brother = node.parent.right
                if (node_brother.left is None or node_brother.left.color == 0) and (node_brother.right is None or node_brother.right.color == 0):
                    node_brother.color = 1
                    node = node.parent
                else:
                    if node_brother.left is None or node_brother.left.color == 0:
                        node_brother.color = 1
                        node_brother.right.color = 0
                        self.left_rotate(node_brother)
                        node_brother = node.parent.left
                    node_brother.color = node.parent.color
                    node.parent.color = 0
                    node_brother.left.color = 0
                    self.right_rotate(node.parent)
                    node = self.root
        node.color = 0

    def inorder(self):
        if False:
            for i in range(10):
                print('nop')
        res = []
        if not self.root:
            return res
        stack = []
        root = self.root
        while root or stack:
            while root:
                stack.append(root)
                root = root.left
            root = stack.pop()
            res.append({'val': root.val, 'color': root.color})
            root = root.right
        return res
if __name__ == '__main__':
    rb = RBTree()
    children = [11, 2, 14, 1, 7, 15, 5, 8, 4]
    for child in children:
        node = RBNode(child, 1)
        print(child)
        rb.insert(node)
    print(rb.inorder())