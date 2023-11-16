def abaco(lista: list) -> int:
    if False:
        for i in range(10):
            print('nop')
    if len(lista) != 7:
        return None
    else:
        resultado = ''
        for digito in lista:
            if len(digito) > 12:
                return None
            resultado += str(digito.find('-'))
        resultado = int(resultado)
        return resultado
print(abaco(['O---OOOOOOOO', 'OOO---OOOOOO', '---OOOOOOOOO', 'OO---OOOOOOO', 'OOOOOOO---OO', 'OOOOOOOOO---', '---OOOOOOOOO']))