n = 999
check_list = []
for i in range(1, n + 1):
    check_list.append(i)
check_list_ope = ['+', '-', '/', '*']
requeriment = input('Ingresa tu operación:')
lista = requeriment.split(' ')
print(lista)
converted_list = [str(num) if index % 2 != 0 else int(num) for (index, num) in enumerate(lista)]
print(converted_list)

def validacion(converted_list):
    if False:
        i = 10
        return i + 15
    g = 0
    for i in converted_list:
        if type(i) == int and i in check_list:
            g += 1
        elif type(i) == str and i in check_list_ope:
            g += 1
    if g == len(converted_list):
        return True
    else:
        return False
if validacion(converted_list):
    print('True')
else:
    print('False')