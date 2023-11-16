class Solution(object):

    def createBinaryTree(self, descriptions):
        if False:
            print('Hello World!')
        '\n        :type descriptions: List[List[int]]\n        :rtype: Optional[TreeNode]\n        '
        nodes = {}
        children = set()
        for (p, c, l) in descriptions:
            parent = nodes.setdefault(p, TreeNode(p))
            child = nodes.setdefault(c, TreeNode(c))
            if l:
                parent.left = child
            else:
                parent.right = child
            children.add(c)
        return nodes[next((p for p in nodes.iterkeys() if p not in children))]