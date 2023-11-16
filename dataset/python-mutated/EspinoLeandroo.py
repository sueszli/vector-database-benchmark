def triples_pitagoricos(maximo):
    if False:
        for i in range(10):
            print('nop')
    triples = []
    for a in range(1, maximo + 1):
        for b in range(a, maximo + 1):
            c = (a ** 2 + b ** 2) ** 0.5
            if c.is_integer() and c <= maximo:
                triples.append((a, b, int(c)))
    return triples
maximo = 100
resultado = triples_pitagoricos(maximo)
print(resultado)