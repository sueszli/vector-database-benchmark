def busca_caracteres_distintos(s1, s2):
    if False:
        return 10
    caracteres_distintos = []
    for (c1, c2) in zip(s1, s2):
        if c1 != c2:
            caracteres_distintos.append(c2)
    return caracteres_distintos
intentos = 0
while intentos < 3:
    s1 = input('Por favor, ingresa la primera cadena de texto: ')
    s2 = input('Por favor, ingresa la segunda cadena de texto: ')
    if len(s1) != len(s2):
        print('Las cadenas deben tener la misma longitud. Inténtalo de nuevo.')
        intentos += 1
        continue
    caracteres_distintos = busca_caracteres_distintos(s1, s2)
    print('Los caracteres diferentes en las cadenas de texto son:', caracteres_distintos)
    break
if intentos == 3:
    print('Has superado el número máximo de intentos.')