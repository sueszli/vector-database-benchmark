def bruteForce(T: str, p: str) -> int:
    if False:
        print('Hello World!')
    (n, m) = (len(T), len(p))
    (i, j) = (0, 0)
    while i < n and j < m:
        if T[i] == p[j]:
            i += 1
            j += 1
        else:
            i = i - (j - 1)
            j = 0
    if j == m:
        return i - j
    else:
        return -1
print(bruteForce('abcdeabc', 'bcd'))