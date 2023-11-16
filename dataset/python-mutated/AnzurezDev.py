import re

def math_ex(expression):
    if False:
        i = 10
        return i + 15
    match = re.search('^(-?\\d+(\\.\\d+)?\\s+[-+*/%]\\s+)*-?\\d+(\\.\\d+)?$', expression)
    validator = True if match else False
    return validator
test1 = math_ex('5 + 6 / 7 - 4')
test2 = math_ex('5 a 6')
print(test1)
print(test2)