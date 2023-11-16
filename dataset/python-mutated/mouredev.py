def pythagorean_triples(max: int) -> list:
    if False:
        return 10
    triples = []
    for a in range(1, max + 1):
        for b in range(a, max + 1):
            ab_squared = a ** 2 + b ** 2
            c = ab_squared ** 0.5
            if c > max:
                break
            if c.is_integer():
                triples.append((a, b, int(c)))
    return triples
print(pythagorean_triples(9))
print(pythagorean_triples(10))
print(pythagorean_triples(20))