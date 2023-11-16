from random import random

class Climate:

    def __init__(self, temp: int, rainprob: float) -> None:
        if False:
            i = 10
            return i + 15
        self.temp = temp
        self.rainprob = rainprob
        self.rain = False

    def simulate_day(self):
        if False:
            print('Hello World!')
        if random() < self.rainprob:
            self.rain = True
        else:
            self.rain = False
        today = (self.temp, self.rain)
        self.set_rainprob()
        self.set_temp()
        return today

    def set_rainprob(self):
        if False:
            while True:
                i = 10
        if self.temp > 25:
            self.rainprob = min(self.rainprob + 0.2, 1)
        elif self.temp < 5:
            self.rainprob = max(self.rainprob - 0.2, 0)

    def set_temp(self):
        if False:
            i = 10
            return i + 15
        if self.rain:
            self.temp -= 1
        if (x := random()) < 0.1:
            if x < 0.05:
                self.temp -= 2
            else:
                self.temp += 2

    def __repr__(self) -> str:
        if False:
            return 10
        return f'Temperature: {self.temp:3d}ºC | Rain: {self.rain:1} | Rainprob: {self.rainprob:.0%}'

def simulate(days: int, initial_temp: int, initial_rainprob: float):
    if False:
        for i in range(10):
            print('nop')
    climate = Climate(initial_temp, initial_rainprob)
    reports = []
    for day in range(days):
        print(f'[#] Day {day + 1:4d}: {climate}')
        reports.append(climate.simulate_day())
    print(f'[!] Report for {days} days simmulation:\n\t - Maximum Temperature: {max((r[0] for r in reports))}ºC\n\t - Minimum Temperature: {min((r[0] for r in reports))}ºC\n\t - Total Rain Days:     {sum((r[1] for r in reports))}')

def get_initial_conditions():
    if False:
        print('Hello World!')
    while True:
        try:
            days = int(input('[?] Days to simulate?: '))
            if days < 1:
                print('[!] Error: value out of range. Enter an integer greater than 0.')
            else:
                break
        except:
            print('[!] Error: input format is not allowed. You must enter an integer.')
    while True:
        try:
            temp = int(input('[?] Initial temperature (ºC)?: '))
            break
        except:
            print('[!] Error: input format is not allowed. You must enter an integer.')
    while True:
        try:
            prob = float(input('[?] Initial rain probability (0-1)?: '))
            if not 0 < prob < 1:
                print('[!] Error: probability out of range. Enter a probability between 0 and 1.')
            else:
                break
        except:
            print('[!] Error: input format is not allowed. You must enter a float number.')
    return (days, temp, prob)
if __name__ == '__main__':
    initial_conditions = get_initial_conditions()
    simulate(*initial_conditions)