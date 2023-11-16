class Solution(object):

    def binaryTreePaths(self, root):
        if False:
            for i in range(10):
                print('nop')
        (result, path) = ([], [])
        self.binaryTreePathsRecu(root, path, result)
        return result

    def binaryTreePathsRecu(self, node, path, result):
        if False:
            return 10
        if node is None:
            return
        if node.left is node.right is None:
            ans = ''
            for n in path:
                ans += str(n.val) + '->'
            result.append(ans + str(node.val))
        if node.left:
            path.append(node)
            self.binaryTreePathsRecu(node.left, path, result)
            path.pop()
        if node.right:
            path.append(node)
            self.binaryTreePathsRecu(node.right, path, result)
            path.pop()