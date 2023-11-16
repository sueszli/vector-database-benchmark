"""
Given a matrix of words and a list of words to search,
return a list of words that exists in the board
This is Word Search II on LeetCode

board = [
         ['o','a','a','n'],
         ['e','t','a','e'],
         ['i','h','k','r'],
         ['i','f','l','v']
         ]

words = ["oath","pea","eat","rain"]
"""

def find_words(board, words):
    if False:
        i = 10
        return i + 15

    def backtrack(board, i, j, trie, pre, used, result):
        if False:
            for i in range(10):
                print('nop')
        '\n        backtrack tries to build each words from\n        the board and return all words found\n\n        @param: board, the passed in board of characters\n        @param: i, the row index\n        @param: j, the column index\n        @param: trie, a trie of the passed in words\n        @param: pre, a buffer of currently build string that differs\n                by recursion stack\n        @param: used, a replica of the board except in booleans\n                to state whether a character has been used\n        @param: result, the resulting set that contains all words found\n\n        @return: list of words found\n        '
        if '#' in trie:
            result.add(pre)
        if i < 0 or i >= len(board) or j < 0 or (j >= len(board[0])):
            return
        if not used[i][j] and board[i][j] in trie:
            used[i][j] = True
            backtrack(board, i + 1, j, trie[board[i][j]], pre + board[i][j], used, result)
            backtrack(board, i, j + 1, trie[board[i][j]], pre + board[i][j], used, result)
            backtrack(board, i - 1, j, trie[board[i][j]], pre + board[i][j], used, result)
            backtrack(board, i, j - 1, trie[board[i][j]], pre + board[i][j], used, result)
            used[i][j] = False
    trie = {}
    for word in words:
        curr_trie = trie
        for char in word:
            if char not in curr_trie:
                curr_trie[char] = {}
            curr_trie = curr_trie[char]
        curr_trie['#'] = '#'
    result = set()
    used = [[False] * len(board[0]) for _ in range(len(board))]
    for i in range(len(board)):
        for j in range(len(board[0])):
            backtrack(board, i, j, trie, '', used, result)
    return list(result)