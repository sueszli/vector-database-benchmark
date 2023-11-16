def length_longest_path(input):
    if False:
        while True:
            i = 10
    '\n    :type input: str\n    :rtype: int\n    '
    (curr_len, max_len) = (0, 0)
    stack = []
    for s in input.split('\n'):
        print('---------')
        print('<path>:', s)
        depth = s.count('\t')
        print('depth: ', depth)
        print('stack: ', stack)
        print('curlen: ', curr_len)
        while len(stack) > depth:
            curr_len -= stack.pop()
        stack.append(len(s.strip('\t')) + 1)
        curr_len += stack[-1]
        print('stack: ', stack)
        print('curlen: ', curr_len)
        if '.' in s:
            max_len = max(max_len, curr_len - 1)
    return max_len
st = 'dir\n\tsubdir1\n\t\tfile1.ext\n\t\tsubsubdirectory1\n\tsubdir2\n\t\tsubsubdir2\n\t\t\tfile2.ext'
st2 = 'a\n\tb1\n\t\tf1.txt\n\taaaaa\n\t\tf2.txt'
print('path:', st2)
print('answer:', length_longest_path(st2))