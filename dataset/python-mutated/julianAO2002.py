def traductroHacker():
    if False:
        while True:
            i = 10
    diccionario = {'a': '4', 'b': '|3', 'c': '[', 'd': ')', 'e': '3', 'f': '|=', 'g': '&', 'h': '#', 'i': '1', 'j': ',_|', 'k': '>|', 'l': '1', 'm': '/\\/\\', 'n': '^/', 'o': '0', 'p': '|*', 'q': '(_,)', 'r': '12', 's': '5', 't': '7', 'u': '(_)', 'v': '\\/', 'w': '\\/\\/', 'x': '><', 'y': 'j', 'z': '2', '1': 'L', '2': 'R', '3': 'E', '4': 'A', '5': 'S', '6': 'b', '7': 'T', '8': 'B', '9': 'g', '0': 'o'}
    frase = input('frase: ')
    frasehAck = ''
    for n in frase:
        if n in diccionario:
            n = diccionario[n]
            frasehAck += n
        else:
            frasehAck += n
    return frasehAck
print(traductroHacker())