def horspool(T: str, p: str) -> int:
    if False:
        i = 10
        return i + 15
    (n, m) = (len(T), len(p))
    bc_table = generateBadCharTable(p)
    i = 0
    while i <= n - m:
        j = m - 1
        while j > -1 and T[i + j] == p[j]:
            j -= 1
        if j < 0:
            return i
        i += bc_table.get(T[i + m - 1], m)
    return -1

def generateBadCharTable(p: str):
    if False:
        print('Hello World!')
    m = len(p)
    bc_table = dict()
    for i in range(m - 1):
        bc_table[p[i]] = m - 1 - i
    return bc_table
print(horspool('abbcfdddbddcaddebc', 'aaaaa'))
print(horspool('abbcfdddbddcaddebc', 'bcf'))
print(horspool('aaaaa', 'bba'))
print(horspool('mississippi', 'issi'))
print(horspool('ababbbbaaabbbaaa', 'bbbb'))