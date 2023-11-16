def triples_pitagoricos(n):
    if False:
        for i in range(10):
            print('nop')
    resultado = []
    for c in range(1, n + 1):
        for b in range(1, c + 1):
            for a in range(1, b + 1):
                if a ** 2 + b ** 2 == c ** 2:
                    resultado.append([a, b, c])
    return resultado
print(triples_pitagoricos(50))