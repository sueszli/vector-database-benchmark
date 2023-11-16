import random

def nextDay(days, temp, rain):
    if False:
        while True:
            i = 10
    minTemp = 100
    maxTemp = 0
    rainCount = 0
    for i in range(days):
        print('Day: ' + str(i + 1))
        if random.randint(0, 100) < 10:
            if random.randint(0, 100) < 50:
                temp = temp - 2
            else:
                temp = temp + 2
        if temp > 25:
            if rain < 80:
                rain = rain + 20
            else:
                rain = 100
        if temp < 5:
            if rain < 20:
                rain = 0
            else:
                rain = rain - 20
        if rain == 100:
            rainCount = rainCount + 1
            print('Rainy day')
        if temp < minTemp:
            minTemp = temp
        if temp > maxTemp:
            maxTemp = temp
        print('Temperature: ' + str(temp))
        print('Rain probability: ' + str(rain))
        print('')
    print('Rainy days: ' + str(rainCount))
    print('Min temperature: ' + str(minTemp))
    print('Max temperature: ' + str(maxTemp))

def main():
    if False:
        for i in range(10):
            print('nop')
    temp = int(input('Insert temperature: '))
    rain = int(input('Insert rain probability(%): '))
    nextDay(int(input('Insert days: ')), temp, rain)
if __name__ == '__main__':
    main()