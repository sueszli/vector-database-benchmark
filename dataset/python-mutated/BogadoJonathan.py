"""
    * Escribe un programa que muestre por consola (con un print) los
    * números de 1 a 100 (ambos incluidos y con un salto de línea entre
    * cada impresión), sustituyendo los siguientes:
    * - Múltiplos de 3 por la palabra "fizz".
    * - Múltiplos de 5 por la palabra "buzz".
    * - Múltiplos de 3 y de 5 a la vez por la palabra "fizzbuzz".
    
"""

def main():
    if False:
        for i in range(10):
            print('nop')
    multiplo = lambda number, multiplo: number % multiplo == 0
    for i in range(1, 101, 1):
        if multiplo(i, 3) and multiplo(i, 5):
            print('fizzbuzz')
        elif multiplo(i, 3):
            print('fizz')
        elif multiplo(i, 5):
            print('buzz')
        else:
            print(i)
if __name__ == '__main__':
    main()