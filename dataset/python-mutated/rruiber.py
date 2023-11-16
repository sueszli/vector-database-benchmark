def T9(i=input('Escribe tu mensaje en tu teclado T9, solo puedes usar digitos y guiones: ')):
    if False:
        for i in range(10):
            print('nop')
    valores = {'1': ',', '11': '.', '111': '?', '1111': '!', '2': 'a', '22': 'b', '222': 'c', '3': 'd', '33': 'e', '333': 'f', '4': 'g', '44': 'h', '444': 'i', '5': 'j', '55': 'k', '555': 'l', '6': 'm', '66': 'n', '666': 'o', '7': 'p', '77': 'q', '777': 'r', '7777': 's', '8': 't', '88': 'u', '888': 'v', '9': 'w', '99': 'x', '999': 'y', '9999': 'z', '-': {}}
    ultimo = '-'
    i = i + ultimo
    mensaje_inicial = []
    mensaje_real = []
    read = []
    for j in i:
        if j not in valores:
            continue
        elif j != '-':
            read.append(j)
        if j == '-':
            read = ''.join(read)
            mensaje_inicial.append(read)
            read = []
    for i in range(len(mensaje_inicial)):
        if mensaje_inicial[i] in valores:
            mensaje_real.append(mensaje_inicial[i])
    for i in range(len(mensaje_real)):
        mensaje_real[i] = valores[mensaje_real[i]]
    mensaje = ''.join(mensaje_real)
    print(mensaje)
T9()