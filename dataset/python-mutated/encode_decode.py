""" Design an algorithm to encode a list of strings to a string.
 The encoded mystring is then sent over the network and is decoded
 back to the original list of strings.
"""

def encode(strs):
    if False:
        while True:
            i = 10
    'Encodes a list of strings to a single string.\n    :type strs: List[str]\n    :rtype: str\n    '
    res = ''
    for string in strs.split():
        res += str(len(string)) + ':' + string
    return res

def decode(s):
    if False:
        return 10
    'Decodes a single string to a list of strings.\n    :type s: str\n    :rtype: List[str]\n    '
    strs = []
    i = 0
    while i < len(s):
        index = s.find(':', i)
        size = int(s[i:index])
        strs.append(s[index + 1:index + 1 + size])
        i = index + 1 + size
    return strs