import time
'\n/*\n * Crea una función que reciba dos parámetros para crear una cuenta atrás.\n * - El primero, representa el número en el que comienza la cuenta.\n * - El segundo, los segundos que tienen que transcurrir entre cada cuenta.\n * - Sólo se aceptan números enteros positivos.\n * - El programa finaliza al llegar a cero.\n * - Debes imprimir cada número de la cuenta atrás.\n */\n'

def cuenta_atras(numero, segundos):
    if False:
        while True:
            i = 10
    while numero >= 0:
        print(numero)
        time.sleep(segundos)
        numero -= 1
if __name__ == '__main__':
    cuenta_atras(10, 3)