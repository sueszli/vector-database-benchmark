import random

def getInitialConditions():
    if False:
        while True:
            i = 10
    try:
        initialTemperature = int(input('Introduce la temperatura inicial (ºC): '))
        initialRainChance = int(input('Introduce la probabilidad de lluvia inicial: '))
        days_prediction = int(input('Introduce el número de días de predicción: '))
        return (initialTemperature, initialRainChance, days_prediction)
    except ValueError:
        print('Introduce valores correctos.')
        getInitialConditions()

def temperatureChange():
    if False:
        i = 10
        return i + 15
    rand = random.randint(0, 100)
    if rand >= 0 and rand <= 10:
        return True
    return False

def temperatureIncrease2Grades():
    if False:
        while True:
            i = 10
    rand = random.choice([0, 1])
    if rand == 0:
        return True
    return False

def printResults(rainyDays, minTemperature, maxTemperature):
    if False:
        while True:
            i = 10
    print('\n')
    print('                      RESULTADOS                    ')
    print('----------------------------------------------------')
    print(f'Días de lluvia: {rainyDays}')
    print(f'Temperatura mínima: {minTemperature} ºC')
    print(f'Temperatura máxima: {maxTemperature} ºC')

def calculatePredictions():
    if False:
        print('Hello World!')
    (initialTemperature, initialRainChance, days_prediction) = getInitialConditions()
    rainyDays = 0
    maxTemperature = initialTemperature
    minTemperature = initialTemperature
    for day in range(1, days_prediction + 1):
        if temperatureChange():
            if temperatureIncrease2Grades():
                initialTemperature += 2
            else:
                initialTemperature -= 2
        if initialTemperature > 25:
            initialRainChance += 20
        if initialTemperature < 5:
            initialRainChance -= 20
        if initialRainChance >= 100:
            initialTemperature -= 1
            rainyDays += 1
        if initialTemperature < minTemperature:
            minTemperature = initialTemperature
        if initialTemperature > maxTemperature:
            maxTemperature = initialTemperature
    return printResults(rainyDays, minTemperature, maxTemperature)
if __name__ == '__main__':
    calculatePredictions()