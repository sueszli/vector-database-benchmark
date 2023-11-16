def find_sum(lista, target):
    if False:
        for i in range(10):
            print('nop')
    combinaciones = []
    encontrado = []
    listaCopy = []
    suma = 0
    lista.sort()
    for (i, v) in enumerate(lista):
        suma = sum(lista)
        combinaciones.append(lista) if suma == target else None
        listaCopy = lista.copy()
        listaCopy.pop(i)
        encontrado = find_sum(listaCopy, target)
        if encontrado != []:
            for elemento in encontrado:
                if elemento not in combinaciones:
                    combinaciones.append(elemento)
    return combinaciones
lista = [1, 5, 3, 2]
lista2 = [1, 2, 2, 1, 3, 4, -1]
lista3 = [1, 7]
print(find_sum(lista, 6))
print(find_sum(lista2, 4))
print(find_sum(lista3, -1))