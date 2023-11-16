abaco = ['O---OOOOOOOO', 'OOO---OOOOOO', '---OOOOOOOOO', 'OO---OOOOOOO', 'OOOOOOO---OO', 'OOOOOOOOO---', '---OOOOOOOOO']

def convertir(abaco):
    if False:
        print('Hello World!')
    multiplicador = 1000000
    num = 0
    for linea in abaco:
        separados = linea.split('---')
        unidades = len(separados[0])
        num += unidades * multiplicador
        multiplicador /= 10
    return num