from math import fmod
"\nThe function `fizzbuzz` prints numbers from 1 to 100, replacing multiples of 3 with 'fizz',\nmultiples of 5 with 'buzz', and multiples of both 3 and 5 with 'fizzbuzz'.\n"

def fizzbuzz() -> None:
    if False:
        while True:
            i = 10
    for number in range(1, 101):
        if fmod(number, 3) == 0 and fmod(number, 5) == 0:
            print('fizzbuzz')
        elif fmod(number, 3) == 0:
            print('fizz')
        elif fmod(number, 5) == 0:
            print('buzz')
        else:
            print(number)
fizzbuzz()