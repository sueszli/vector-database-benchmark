def sunday(T: str, p: str) -> int:
    if False:
        while True:
            i = 10
    (n, m) = (len(T), len(p))
    bc_table = generateBadCharTable(p)
    i = 0
    while i <= n - m:
        j = 0
        if T[i:i + m] == p:
            return i
        if i + m >= n:
            return -1
        i += bc_table.get(T[i + m], m + 1)
    return -1

def generateBadCharTable(p: str):
    if False:
        i = 10
        return i + 15
    m = len(p)
    bc_table = dict()
    for i in range(m):
        bc_table[p[i]] = m - i
    return bc_table
print(sunday('abbcfdddbddcaddebc', 'aaaaa'))
print(sunday('abbcfdddbddcaddebc', 'bcf'))
print(sunday('aaaaa', 'bba'))
print(sunday('mississippi', 'issi'))
print(sunday('ababbbbaaabbbaaa', 'bbbb'))