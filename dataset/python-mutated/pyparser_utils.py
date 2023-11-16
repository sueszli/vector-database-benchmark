"""PyParser-related utilities.

This module collects various utilities related to the parse trees produced by
the pyparser.

  GetLogicalLine: produces a list of tokens from the logical lines within a
    range.
  GetTokensInSubRange: produces a sublist of tokens from a current token list
    within a range.
  GetTokenIndex: Get the index of a token.
  GetNextTokenIndex: Get the index of the next token after a given position.
  GetPrevTokenIndex: Get the index of the previous token before a given
    position.
  TokenStart: Convenience function to return the token's start as a tuple.
  TokenEnd: Convenience function to return the token's end as a tuple.
"""

def GetLogicalLine(logical_lines, node):
    if False:
        for i in range(10):
            print('nop')
    "Get a list of tokens within the node's range from the logical lines."
    start = TokenStart(node)
    end = TokenEnd(node)
    tokens = []
    for line in logical_lines:
        if line.start > end:
            break
        if line.start <= start or line.end >= end:
            tokens.extend(GetTokensInSubRange(line.tokens, node))
    return tokens

def GetTokensInSubRange(tokens, node):
    if False:
        while True:
            i = 10
    'Get a subset of tokens representing the node.'
    start = TokenStart(node)
    end = TokenEnd(node)
    tokens_in_range = []
    for tok in tokens:
        tok_range = (tok.lineno, tok.column)
        if tok_range >= start and tok_range < end:
            tokens_in_range.append(tok)
    return tokens_in_range

def GetTokenIndex(tokens, pos):
    if False:
        for i in range(10):
            print('nop')
    'Get the index of the token at pos.'
    for (index, token) in enumerate(tokens):
        if (token.lineno, token.column) == pos:
            return index
    return None

def GetNextTokenIndex(tokens, pos):
    if False:
        while True:
            i = 10
    'Get the index of the next token after pos.'
    for (index, token) in enumerate(tokens):
        if (token.lineno, token.column) >= pos:
            return index
    return None

def GetPrevTokenIndex(tokens, pos):
    if False:
        for i in range(10):
            print('nop')
    'Get the index of the previous token before pos.'
    for (index, token) in enumerate(tokens):
        if index > 0 and (token.lineno, token.column) >= pos:
            return index - 1
    return None

def TokenStart(node):
    if False:
        i = 10
        return i + 15
    return (node.lineno, node.col_offset)

def TokenEnd(node):
    if False:
        return 10
    return (node.end_lineno, node.end_col_offset)

def AstDump(node):
    if False:
        return 10
    import ast
    print(ast.dump(node, include_attributes=True, indent=4))