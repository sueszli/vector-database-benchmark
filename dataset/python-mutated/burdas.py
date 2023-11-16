import math

def find_tri_pitagoras(max: int) -> list:
    if False:
        print('Hello World!')
    result = []

    def calcular_catetos(hipotenusa):
        if False:
            return 10
        for a in range(1, hipotenusa):
            b = math.sqrt(hipotenusa ** 2 - a ** 2)
            if b.is_integer():
                return [a, int(b)]
        return None
    for a in range(1, max + 1):
        b = calcular_catetos(a)
        if b is not None:
            result.append([b[0], b[1], a])
    return result
print(find_tri_pitagoras(10))