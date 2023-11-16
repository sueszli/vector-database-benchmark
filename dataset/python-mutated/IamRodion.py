import time, os

def limpiarPantalla():
    if False:
        while True:
            i = 10
    if os.name == 'posix':
        os.system('clear')
    else:
        os.system('cls')

def generarSemilla():
    if False:
        i = 10
        return i + 15
    semilla = time.time_ns()
    return semilla

def generarEnteroPseudoAleatorio():
    if False:
        i = 10
        return i + 15
    semilla = generarSemilla()
    número = semilla % 101
    return número

def comprobarAleatoriedad(función, ciclos):
    if False:
        print('Hello World!')
    númerosGenerados = []
    for i in range(ciclos):
        númerosGenerados.append(función())
    for i in range(101):
        porcentaje = númerosGenerados.count(i) * 100 / ciclos
        print(f'Número {i} aparece {númerosGenerados.count(i)} veces, ocupando el {porcentaje}% del total')

def main():
    if False:
        while True:
            i = 10
    limpiarPantalla()
    print(f'Número generado: {generarEnteroPseudoAleatorio()}')
    input("[!] Presione 'Enter' para comprobar aleatoriedad de la función: ")
    limpiarPantalla()
    comprobarAleatoriedad(generarEnteroPseudoAleatorio, ciclos=100)
if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\nCerrando programa')