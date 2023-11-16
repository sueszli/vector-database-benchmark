def boyerMoore(T: str, p: str) -> int:
    if False:
        i = 10
        return i + 15
    (n, m) = (len(T), len(p))
    bc_table = generateBadCharTable(p)
    gs_list = generageGoodSuffixList(p)
    i = 0
    while i <= n - m:
        j = m - 1
        while j > -1 and T[i + j] == p[j]:
            j -= 1
        if j < 0:
            return i
        bad_move = j - bc_table.get(T[i + j], -1)
        good_move = gs_list[j]
        i += max(bad_move, good_move)
    return -1

def generateBadCharTable(p: str):
    if False:
        i = 10
        return i + 15
    bc_table = dict()
    for i in range(len(p)):
        bc_table[p[i]] = i
    return bc_table

def generageGoodSuffixList(p: str):
    if False:
        i = 10
        return i + 15
    m = len(p)
    gs_list = [m for _ in range(m)]
    suffix = generageSuffixArray(p)
    j = 0
    for i in range(m - 1, -1, -1):
        if suffix[i] == i + 1:
            while j < m - 1 - i:
                if gs_list[j] == m:
                    gs_list[j] = m - 1 - i
                j += 1
    for i in range(m - 1):
        gs_list[m - 1 - suffix[i]] = m - 1 - i
    return gs_list

def generageSuffixArray(p: str):
    if False:
        return 10
    m = len(p)
    suffix = [m for _ in range(m)]
    for i in range(m - 2, -1, -1):
        start = i
        while start >= 0 and p[start] == p[m - 1 - i + start]:
            start -= 1
        suffix[i] = i - start
    return suffix
print(boyerMoore('abbcfdddbddcaddebc', 'aaaaa'))
print(boyerMoore('', ''))