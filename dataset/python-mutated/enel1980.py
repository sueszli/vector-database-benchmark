import random

def meteorologo() -> list:
    if False:
        i = 10
        return i + 15
    final = [['D, T, P']]
    temp = int(input('Ingrese temperatura inicial: '))
    prob_lluvia = int(input('Ingrese probabilidad de lluvia inicial: '))
    dias = int(input('¿Cuántos días vamos a estimar: '))
    for i in range(1, dias + 1):
        if prob_lluvia == 100:
            temp -= 1
        if random.randint(0, 100) < 10:
            temp += 2
        if random.randint(0, 100) > 90:
            temp -= 2
        if temp > 25:
            prob_lluvia += 20
            if prob_lluvia > 99:
                prob_lluvia = 100
        elif temp < 5:
            prob_lluvia -= 20
            if prob_lluvia < 1:
                prob_lluvia = 0
        print(f'La temperatura el dia {i} es: {temp} Celsius')
        print(f'La probabilidad de lluvia el dia {i} es: {prob_lluvia} %')
        final.append([f'{i}', f'{temp}', f'{prob_lluvia}'])
    return final
final = meteorologo()
for i in range(len(final)):
    print(final[i])