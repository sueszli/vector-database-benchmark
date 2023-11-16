def LenguajeHacker(string):
    if False:
        i = 10
        return i + 15
    leetString = []
    leetAlphabet = ['4', 'I3', '[', ')', '3', '|=', '&', '#', '1', ',_|', '>|', '1', '/\\/\\', '^/', '0', '|*', '(_,)', 'I2', '5', '7', '(_)', '\\/', '\\/\\/', '><', 'j', '2', 'L', 'R', 'E', 'A', 'S', 'b', 'T', 'B', 'g', 'o', ' ']
    alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '1', '2', '3', '4', '5', '6', '7', '8', '9', '0', ' ']
    for char in string.lower():
        alphabet.index(char)
        leetString.append(char.replace(char, leetAlphabet[alphabet.index(char)]))
    print('Frase original: ' + string)
    print('Frase en lenguaje hacker: ' + ''.join(leetString))
LenguajeHacker('Hola Mouredev')
LenguajeHacker('Reto 1 Completado')