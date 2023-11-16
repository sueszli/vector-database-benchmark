def numero_columna(columna):
    if False:
        return 10
    alfabeto = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    return sum([(alfabeto.index(columna[n]) + 1) * 26 ** (len(columna) - 1 - n) for n in range(len(columna))])
print(numero_columna('CA'))