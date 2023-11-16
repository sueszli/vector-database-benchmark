def cifrar_cesar(texto, desplazamiento):
    if False:
        i = 10
        return i + 15
    resultado = ''
    for letra in texto:
        if letra.isalpha():
            codigo = ord(letra)
            codigo_cifrado = (codigo - ord('a') + desplazamiento) % 26 + ord('a')
            letra_cifrada = chr(codigo_cifrado)
            resultado += letra_cifrada
        else:
            resultado += letra
    return resultado

def descifrar_cesar(texto_cifrado, desplazamiento):
    if False:
        for i in range(10):
            print('nop')
    resultado = ''
    for letra in texto_cifrado:
        if letra.isalpha():
            codigo = ord(letra)
            codigo_descifrado = (codigo - ord('a') - desplazamiento) % 26 + ord('a')
            letra_descifrada = chr(codigo_descifrado)
            resultado += letra_descifrada
        else:
            resultado += letra
    return resultado
texto_original = 'hola mundo'
desplazamiento = 3
texto_cifrado = cifrar_cesar(texto_original, desplazamiento)
print('Texto cifrado:', texto_cifrado)
texto_descifrado = descifrar_cesar(texto_cifrado, desplazamiento)
print('Texto descifrado:', texto_descifrado)