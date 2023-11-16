import re
WHITESPACE_NORMALIZER = re.compile('\\s+')
SPACE = chr(32)
SPACE_ESCAPE = chr(9601)
PRINTABLE_LATIN = set(list(range(32, 126 + 1)) + list(range(161, 172 + 1)) + list(range(174, 255 + 1)))
BYTE_TO_BCHAR = {b: chr(b) if b in PRINTABLE_LATIN else chr(256 + b) for b in range(256)}
BCHAR_TO_BYTE = {bc: b for (b, bc) in BYTE_TO_BCHAR.items()}

def byte_encode(x: str) -> str:
    if False:
        print('Hello World!')
    normalized = WHITESPACE_NORMALIZER.sub(SPACE, x)
    return ''.join([BYTE_TO_BCHAR[b] for b in normalized.encode('utf-8')])

def byte_decode(x: str) -> str:
    if False:
        for i in range(10):
            print('nop')
    try:
        return bytes([BCHAR_TO_BYTE[bc] for bc in x]).decode('utf-8')
    except ValueError:
        return ''

def smart_byte_decode(x: str) -> str:
    if False:
        i = 10
        return i + 15
    output = byte_decode(x)
    if output == '':
        n_bytes = len(x)
        f = [0 for _ in range(n_bytes + 1)]
        pt = [0 for _ in range(n_bytes + 1)]
        for i in range(1, n_bytes + 1):
            (f[i], pt[i]) = (f[i - 1], i - 1)
            for j in range(1, min(4, i) + 1):
                if f[i - j] + 1 > f[i] and len(byte_decode(x[i - j:i])) > 0:
                    (f[i], pt[i]) = (f[i - j] + 1, i - j)
        cur_pt = n_bytes
        while cur_pt > 0:
            if f[cur_pt] == f[pt[cur_pt]] + 1:
                output = byte_decode(x[pt[cur_pt]:cur_pt]) + output
            cur_pt = pt[cur_pt]
    return output