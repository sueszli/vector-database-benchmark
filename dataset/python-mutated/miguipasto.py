def tripletes_pitagoricos(max: int):
    if False:
        for i in range(10):
            print('nop')
    tripletes = []
    for a in range(1, max + 1):
        for b in range(a, max + 1):
            c_cuadrado = a ** 2 + b ** 2
            c = int(c_cuadrado ** 0.5)
            if c <= max and c_cuadrado == c ** 2:
                tripletes.append((a, b, c))
    return tripletes
resultados = tripletes_pitagoricos(10)
print('Los tripletes pitagóricos son:', resultados)