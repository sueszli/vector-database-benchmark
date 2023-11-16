def except_handler(self, lhs, n, rule, ast, tokens, first, last):
    if False:
        return 10
    end_token = tokens[last - 1]
    if self.version[:2] == (1, 4):
        return False
    if end_token != 'COME_FROM':
        return False
    return end_token.attr < tokens[first].offset