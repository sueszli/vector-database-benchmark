def kmp(T: str, p: str) -> int:
    if False:
        for i in range(10):
            print('nop')
    (n, m) = (len(T), len(p))
    next = generateNext(p)
    j = 0
    for i in range(n):
        while j > 0 and T[i] != p[j]:
            j = next[j - 1]
        if T[i] == p[j]:
            j += 1
        if j == m:
            return i - j + 1
    return -1

def generateNext(p: str):
    if False:
        print('Hello World!')
    m = len(p)
    next = [0 for _ in range(m)]
    left = 0
    for right in range(1, m):
        while left > 0 and p[left] != p[right]:
            left = next[left - 1]
        if p[left] == p[right]:
            left += 1
        next[right] = left
    return next
print(kmp('abbcfdddbddcaddebc', 'ABCABCD'))
print(kmp('abbcfdddbddcaddebc', 'bcf'))
print(kmp('aaaaa', 'bba'))
print(kmp('mississippi', 'issi'))
print(kmp('ababbbbaaabbbaaa', 'bbbb'))