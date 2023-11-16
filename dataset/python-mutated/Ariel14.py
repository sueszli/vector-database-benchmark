def combinaciones_posibles(lista_numeros):
    if False:
        for i in range(10):
            print('nop')
    return combinaciones_posibles_recursivas([], lista_numeros)

def combinaciones_posibles_recursivas(actual, lista_numeros):
    if False:
        for i in range(10):
            print('nop')
    if lista_numeros != []:
        return combinaciones_posibles_recursivas(actual, lista_numeros[1:]) + combinaciones_posibles_recursivas(actual + [lista_numeros[0]], lista_numeros[1:])
    return [actual]

def filtrar_segun_sumatoria(lista_de_combinaciones, valor):
    if False:
        i = 10
        return i + 15
    lista_filtrada = []
    for i in lista_de_combinaciones:
        if sum(i) == valor:
            lista_filtrada.append(i)
    return lista_filtrada
lista_numeros = [1, 5, 3, 2]
valor_objetivo = 6
combinaciones = combinaciones_posibles(lista_numeros)
resultado = filtrar_segun_sumatoria(combinaciones, valor_objetivo)
print(resultado)