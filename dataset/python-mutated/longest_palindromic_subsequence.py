def longest_palindromic_subsequence(s):
    if False:
        return 10
    k = len(s)
    olist = [0] * k
    nList = [0] * k
    logestSubStr = ''
    logestLen = 0
    for j in range(0, k):
        for i in range(0, j + 1):
            if j - i <= 1:
                if s[i] == s[j]:
                    nList[i] = 1
                    len_t = j - i + 1
                    if logestLen < len_t:
                        logestSubStr = s[i:j + 1]
                        logestLen = len_t
            elif s[i] == s[j] and olist[i + 1]:
                nList[i] = 1
                len_t = j - i + 1
                if logestLen < len_t:
                    logestSubStr = s[i:j + 1]
                    logestLen = len_t
        olist = nList
        nList = [0] * k
    return logestLen