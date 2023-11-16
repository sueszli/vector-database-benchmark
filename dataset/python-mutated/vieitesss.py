"""
 Crea una función que encuentre todos los triples pitagóricos
 (ternas) menores o iguales a un número dado.
 - Debes buscar información sobre qué es un triple pitagórico.
 - La función únicamente recibe el número máximo que puede
   aparecer en el triple.
 - Ejemplo: Los triples menores o iguales a 10 están
   formados por (3, 4, 5) y (6, 8, 10).
"""

def get_pythagorean_triples(max_n: int) -> list:
    if False:
        return 10
    'Devuelve los triples pitagóricos con valores menores al número dado\n\n    Args:\n        max_n (int): valor máximo posible dentro de un triple pitagórico\n\n    Returns:\n        list: lista de triples pitagóricos válidos\n    '
    triples = []
    stop = False
    for i in range(3, max_n):
        for j in range(i + 1, max_n):
            result = (i ** 2 + j ** 2) ** 0.5
            if result > max_n:
                if j == i + 1:
                    stop = True
                break
            if result % int(result) == 0:
                triples.append((i, j, int(result)))
        if stop:
            break
    return triples
p_triples = get_pythagorean_triples(20)
print(p_triples)
print(len(p_triples))