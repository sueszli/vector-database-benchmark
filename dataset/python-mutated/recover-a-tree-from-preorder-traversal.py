class TreeNode(object):

    def __init__(self, x):
        if False:
            for i in range(10):
                print('nop')
        self.val = x
        self.left = None
        self.right = None

class Solution(object):

    def recoverFromPreorder(self, S):
        if False:
            print('Hello World!')
        '\n        :type S: str\n        :rtype: TreeNode\n        '
        i = 0
        stack = []
        while i < len(S):
            level = 0
            while i < len(S) and S[i] == '-':
                level += 1
                i += 1
            while len(stack) > level:
                stack.pop()
            val = []
            while i < len(S) and S[i] != '-':
                val.append(S[i])
                i += 1
            node = TreeNode(int(''.join(val)))
            if stack:
                if stack[-1].left is None:
                    stack[-1].left = node
                else:
                    stack[-1].right = node
            stack.append(node)
        return stack[0]

class Solution2(object):

    def recoverFromPreorder(self, S):
        if False:
            i = 10
            return i + 15
        '\n        :type S: str\n        :rtype: TreeNode\n        '

        def recoverFromPreorderHelper(S, level, i):
            if False:
                print('Hello World!')
            j = i[0]
            while j < len(S) and S[j] == '-':
                j += 1
            if level != j - i[0]:
                return None
            i[0] = j
            while j < len(S) and S[j] != '-':
                j += 1
            node = TreeNode(int(S[i[0]:j]))
            i[0] = j
            node.left = recoverFromPreorderHelper(S, level + 1, i)
            node.right = recoverFromPreorderHelper(S, level + 1, i)
            return node
        return recoverFromPreorderHelper(S, 0, [0])