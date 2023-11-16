def decode_string(s):
    if False:
        while True:
            i = 10
    '\n    :type s: str\n    :rtype: str\n    '
    stack = []
    cur_num = 0
    cur_string = ''
    for c in s:
        if c == '[':
            stack.append((cur_string, cur_num))
            cur_string = ''
            cur_num = 0
        elif c == ']':
            (prev_string, num) = stack.pop()
            cur_string = prev_string + num * cur_string
        elif c.isdigit():
            cur_num = cur_num * 10 + int(c)
        else:
            cur_string += c
    return cur_string