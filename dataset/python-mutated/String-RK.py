def rabinKarp(T: str, p: str, d, q) -> int:
    if False:
        while True:
            i = 10
    (n, m) = (len(T), len(p))
    if n < m:
        return -1
    (hash_p, hash_t) = (0, 0)
    for i in range(m):
        hash_p = (hash_p * d + ord(p[i])) % q
        hash_t = (hash_t * d + ord(T[i])) % q
    power = pow(d, m - 1) % q
    for i in range(n - m + 1):
        if hash_p == hash_t:
            match = True
            for j in range(m):
                if T[i + j] != p[j]:
                    match = False
                    break
            if match:
                return i
        if i < n - m:
            hash_t = (hash_t - power * ord(T[i])) % q
            hash_t = (hash_t * d + ord(T[i + m])) % q
            hash_t = (hash_t + q) % q
    return -1
print(rabinKarp('aaaaa', 'bba', 256, 101))