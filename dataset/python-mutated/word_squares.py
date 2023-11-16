import collections

def word_squares(words):
    if False:
        return 10
    n = len(words[0])
    fulls = collections.defaultdict(list)
    for word in words:
        for i in range(n):
            fulls[word[:i]].append(word)

    def build(square):
        if False:
            print('Hello World!')
        if len(square) == n:
            squares.append(square)
            return
        prefix = ''
        for k in range(len(square)):
            prefix += square[k][len(square)]
        for word in fulls[prefix]:
            build(square + [word])
    squares = []
    for word in words:
        build([word])
    return squares