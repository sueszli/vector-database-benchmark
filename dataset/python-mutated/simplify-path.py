class Solution(object):

    def simplifyPath(self, path):
        if False:
            return 10
        (stack, tokens) = ([], path.split('/'))
        for token in tokens:
            if token == '..' and stack:
                stack.pop()
            elif token != '..' and token != '.' and token:
                stack.append(token)
        return '/' + '/'.join(stack)