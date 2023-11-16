"""
 * Crea una función que encuentre todos los triples pitagóricos
 * (ternas) menores o iguales a un número dado.
 * - Debes buscar información sobre qué es un triple pitagórico.
 * - La función únicamente recibe el número máximo que puede
 *   aparecer en el triple.
 * - Ejemplo: Los triples menores o iguales a 10 están
 *   formados por (3, 4, 5) y (6, 8, 10).
"""

def pitagoras(n):
    if False:
        print('Hello World!')
    print('Los triples menores o iguales a', n, 'son:')
    for a in range(1, n + 1):
        for b in range(a, n + 1):
            cCuadrado = a ** 2 + b ** 2
            c = int(cCuadrado ** 0.5)
            if c ** 2 == cCuadrado and c <= n:
                print('(', a, ',', b, ',', c, ')')
pitagoras(20)
'\n3,4,5\n5,12,13\n6,8,10\n8,15,17\n9,12,15\n12,16,20\n'
pitagoras(10)
'\n3,4,5\n6,8,10\n'
pitagoras(5)
'\n3,4,5\n'