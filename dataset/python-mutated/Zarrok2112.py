import random

def getStartInputs():
    if False:
        i = 10
        return i + 15
    while True:
        try:
            initialTemperature = int(input('Introduce la temperatura inicial (ºC): '))
            if -50 <= initialTemperature <= 60:
                break
            else:
                print('La temperatura inicial debe estar entre -50 y 60.')
        except ValueError:
            print('La temperatura inicial debe ser un número entero.')
    while True:
        try:
            initialRainChance = int(input('Introduce la probabilidad de lluvia inicial: '))
            if 0 <= initialRainChance <= 100:
                break
            else:
                print('La probabilidad de lluvia inicial debe estar entre 0 y 100.')
        except ValueError:
            print('La probabilidad de lluvia inicial debe ser un número entero.')
    while True:
        try:
            days_prediction = int(input('Introduce el número de días de predicción: '))
            if days_prediction >= 0:
                break
            else:
                print('El número de días de predicción debe ser un número entero no negativo.')
        except ValueError:
            print('El número de días de predicción debe ser un número entero.')
    return (initialTemperature, initialRainChance, days_prediction)

def temperatureChange():
    if False:
        while True:
            i = 10
    rand = random.randint(0, 100)
    if rand >= 0 and rand <= 10:
        rand = random.choice([0, 1])
        if rand == 0:
            return 2
        return -2
    return False

def oracule_app():
    if False:
        return 10
    (initialTemperature, initialRainChance, days_prediction) = getStartInputs()
    rainyDays = 0
    temperature = initialTemperature
    rainChance = initialRainChance
    minTemperature = initialTemperature
    maxTemperature = initialTemperature
    for day in range(1, days_prediction + 1):
        temperatureChangeValue = temperatureChange()
        if temperatureChangeValue:
            temperature += temperatureChangeValue
        if temperature > 25:
            rainChance += 20
        if temperature < 5:
            rainChance -= 20
        if rainChance >= 100:
            rainyDays += 1
            temperature -= 1
        if temperature > maxTemperature:
            maxTemperature = temperature
        if temperature < minTemperature:
            minTemperature = temperature
    print('\n')
    print('                      RESULTADOS                    ')
    print('----------------------------------------------------')
    print(f'Días de lluvia: {rainyDays}')
    print(f'Temperatura mínima: {minTemperature} ºC')
    print(f'Temperatura máxima: {maxTemperature} ºC')
    print('\n')
if __name__ == '__main__':
    oracule_app()