import random

def revword(word):
    if False:
        while True:
            i = 10
    if random.randint(1, 2) == 1:
        return word[::-1]
    return word

def step(word, x, xf, y, yf, grid):
    if False:
        i = 10
        return i + 15
    for i in range(len(word)):
        if grid[xf(i)][yf(i)] != '' and grid[xf(i)][yf(i)] != word[i]:
            return False
    for i in range(len(word)):
        grid[xf(i)][yf(i)] = word[i]
    return True

def check(word, dir, x, y, grid, rows, cols):
    if False:
        i = 10
        return i + 15
    if dir == 1:
        if x - len(word) < 0 or y - len(word) < 0:
            return False
        return step(word, x, lambda i: x - i, y, lambda i: y - i, grid)
    elif dir == 2:
        if x - len(word) < 0:
            return False
        return step(word, x, lambda i: x - i, y, lambda i: y, grid)
    elif dir == 3:
        if x - len(word) < 0 or y + (len(word) - 1) >= cols:
            return False
        return step(word, x, lambda i: x - i, y, lambda i: y + i, grid)
    elif dir == 4:
        if y - len(word) < 0:
            return False
        return step(word, x, lambda i: x, y, lambda i: y - i, grid)

def wordfinder(words, rows=20, cols=20, attempts=50, alph='ABCDEFGHIJKLMNOPQRSTUVWXYZ'):
    if False:
        return 10
    '\n    Attempt to arrange words into a letter-grid with the specified\n    number of rows and columns.  Try each word in several positions\n    and directions, until it can be fitted into the grid, or the\n    maximum number of allowable attempts is exceeded.  Returns a tuple\n    consisting of the grid and the words that were successfully\n    placed.\n\n    :param words: the list of words to be put into the grid\n    :type words: list\n    :param rows: the number of rows in the grid\n    :type rows: int\n    :param cols: the number of columns in the grid\n    :type cols: int\n    :param attempts: the number of times to attempt placing a word\n    :type attempts: int\n    :param alph: the alphabet, to be used for filling blank cells\n    :type alph: list\n    :rtype: tuple\n    '
    words = sorted(words, key=len, reverse=True)
    grid = []
    used = []
    for i in range(rows):
        grid.append([''] * cols)
    for word in words:
        word = word.strip().upper()
        save = word
        word = revword(word)
        for attempt in range(attempts):
            r = random.randint(0, len(word))
            dir = random.choice([1, 2, 3, 4])
            x = random.randint(0, rows)
            y = random.randint(0, cols)
            if dir == 1:
                x += r
                y += r
            elif dir == 2:
                x += r
            elif dir == 3:
                x += r
                y -= r
            elif dir == 4:
                y += r
            if 0 <= x < rows and 0 <= y < cols:
                if check(word, dir, x, y, grid, rows, cols):
                    used.append(save)
                    break
    for i in range(rows):
        for j in range(cols):
            if grid[i][j] == '':
                grid[i][j] = random.choice(alph)
    return (grid, used)

def word_finder():
    if False:
        for i in range(10):
            print('nop')
    from nltk.corpus import words
    wordlist = words.words()
    random.shuffle(wordlist)
    wordlist = wordlist[:200]
    wordlist = [w for w in wordlist if 3 <= len(w) <= 12]
    (grid, used) = wordfinder(wordlist)
    print('Word Finder\n')
    for i in range(len(grid)):
        for j in range(len(grid[i])):
            print(grid[i][j], end=' ')
        print()
    print()
    for i in range(len(used)):
        print('%d:' % (i + 1), used[i])
if __name__ == '__main__':
    word_finder()