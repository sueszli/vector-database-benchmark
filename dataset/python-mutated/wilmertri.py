def multiplication_table(n: int) -> None:
    if False:
        i = 10
        return i + 15
    for i in range(1, 11):
        print(f'{n} x {i} = {i * n}')
number = int(input('Ingrese un número para imprimir su tabla de multiplicar: '))
multiplication_table(number)