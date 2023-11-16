numbers = str(list(range(0, 10)))
operations = ['+', '-', '*', '/']

def math_expression(digit):
    if False:
        for i in range(10):
            print('nop')
    if digit in operations:
        return True
    else:
        return False

def number_expression(digit):
    if False:
        for i in range(10):
            print('nop')
    if digit in numbers:
        return True
    else:
        return False

def evaluate(expression):
    if False:
        return 10
    number_expression(expression[0])
    for i in expression:
        if math_expression(i) or number_expression(i) or i == ' ':
            continue
        else:
            return False
    return True
expresion1 = '5 + a / 7 - 4'
print(f'La expresion {expresion1} es: {evaluate(expresion1)}')