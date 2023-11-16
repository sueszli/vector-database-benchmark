def print_table_of(num: int | float) -> None:
    if False:
        return 10
    '\n    Dado un número \'num\' (no se especifica si entero, real,\n    positivo o negativo), imprime su "tabla de multiplicar",\n    tal que:\n    num x  1 = ...\n    num x  2 = ...\n    ...\n    num x 10 = ...\n\n    Args:\n        num (int, float): número cuya tabla queremos imprimir.\n\n    Returns:\n        Nada. Imprime directamente.\n    '
    if not isinstance(num, (int, float)):
        print(f'{num} no es un número válido. Ignorando...')
        return
    for i in range(1, 11):
        print(f'{num} x {i:2d} = {num * i}')

def get_number() -> int | float | None:
    if False:
        while True:
            i = 10
    '\n    Solicita un número al usuario. Introducir un valor que no pueda\n    procesarse no eleva un error, sino que ignora e imprime un mensaje.\n\n    Returns:\n        Devuelve el input del usuario, convertido en int o float. Si el usuario\n        desea salir, introducirá una "q", y la función devolverá None.\n    '
    while True:
        number = input('Introduce un número (q = salir): ')
        if number == 'q':
            return None
        try:
            if '.' in number:
                return float(number)
            else:
                return int(number)
        except ValueError:
            print("Ese valor es inválido. Introduce un número, o la letra 'q' para salir.")
if __name__ == '__main__':
    while True:
        input_number = get_number()
        if input_number is None:
            break
        print_table_of(input_number)