def recip_square(i):
    if False:
        return 10
    return 1.0 / i ** 2

def approx_pi(n=10000000):
    if False:
        for i in range(10):
            print('nop')
    val = 0.0
    for k in range(1, n + 1):
        val += recip_square(k)
    return (6 * val) ** 0.5