def cifrado(texto, desplazamiento=3):
    if False:
        return 10
    cifrado = ''
    for caracter in texto:
        if caracter.isalpha():
            codigo = ord(caracter)
            nuevo_codigo = codigo + desplazamiento
            if nuevo_codigo > ord('z'):
                nuevo_codigo -= 26
            elif nuevo_codigo < ord('A'):
                nuevo_codigo += 26
            elif nuevo_codigo > ord('z'):
                nuevo_codigo -= 26
            elif nuevo_codigo < ord('a'):
                nuevo_codigo += 26
            caracter_cifrado = chr(nuevo_codigo)
            cifrado += caracter_cifrado
        else:
            cifrado += caracter
    return cifrado

def descifrado(texto, desplazamiento=3):
    if False:
        while True:
            i = 10
    return cifrado(texto, -desplazamiento)
a = int(input('\n1. cifrar\n2. descifrar\n'))
if a == 1:
    texto = input('type a text')
    print(cifrado(texto))
elif a == 2:
    texto = input('type a text')
    print(descifrado(texto))
else:
    print('type a correct option')