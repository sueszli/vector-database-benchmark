print('🔢 Tabla de multiplicar 🔢')

def main():
    if False:
        i = 10
        return i + 15
    try:
        value: str = input('Ingresa un numero entero:')
        valueInt = int(value)
        operation(valueInt)
    except:
        print('Porfavor ingresa un valor valido, debe ser un numero entero.')
        main()

def operation(number: int):
    if False:
        i = 10
        return i + 15
    print(f'La tabla de multiplicacion de {number} es:')
    for i in range(1, 11):
        print(f'{number} x {i} = {number * i}')
if __name__ == '__main__':
    main()