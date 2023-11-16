def validar(numero):
    if False:
        while True:
            i = 10
    if numero % 2 == 0:
        resultado_par = 'es par.'
    else:
        resultado_par = 'es impar.'
    if numero <= 2:
        resultado_primo = ''
    for num in range(2, numero):
        if numero % num == 0:
            resultado_primo = 'no es primo'
            break
        else:
            resultado_primo = 'es primo'
    (a, b) = (0, 1)
    lista = list()
    while a <= numero - 1:
        (a, b) = (b, a + b)
        lista.append(a)
    if numero in lista:
        resultado_fibo = 'es fibonacci'
    else:
        resultado_fibo = 'no es fibonacci'
    print(f'{numero} {resultado_primo}, {resultado_fibo} y {resultado_par}')
validar(13)