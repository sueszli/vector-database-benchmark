def reverseString(s) -> None:
    if False:
        i = 10
        return i + 15
    left = 0
    right = len(s) - 1

    def recursiveReverse(l, r):
        if False:
            while True:
                i = 10
        if l == r or l > r:
            return
        (s[l], s[r]) = (s[r], s[l])
        return recursiveReverse(l + 1, r - 1)
    recursiveReverse(left, right)
    return s
print(reverseString(['h', 'e', 'l', 'l', 'o']))
print(reverseString(['H', 'a', 'n', 'n', 'a', 'h']))