import re

def checkMathExpression(expresion: str) -> bool:
    if False:
        return 10
    'Crea una función que reciba una expresión matemática (String)\n    y compruebe si es correcta. Retornará true o false.\n    - Para que una expresión matemática sea correcta debe poseer\n    un número, una operación y otro número separados por espacios.\n    Tantos números y operaciones como queramos.\n    - Números positivos, negativos, enteros o decimales.\n    - Operaciones soportadas: + - * / % \n    Ejemplos:\n    "5 + 6 / 7 - 4" -> true\n    "5 a 6" -> false'
    return re.match('[+-]?[0-9]*[.]?[0-9]*\\ [+-\\/\\*]\\ [0-9]*[.]?[0-9]*', expresion) != None
print(checkMathExpression('3 + 5'))
print(checkMathExpression('3 a 5'))
print(checkMathExpression('-3 + 5'))
print(checkMathExpression('- 3 + 5'))
print(checkMathExpression('-3 a 5'))
print(checkMathExpression('-3+5'))
print(checkMathExpression('3 + 5 - 1 / 4 % 8'))