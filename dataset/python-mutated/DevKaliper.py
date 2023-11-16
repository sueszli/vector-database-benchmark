from time import sleep
number_to_count = input('Give me the number wich you want to begin: ')
seconds_to_wait = input('Give the seconds that you want to wait between each number: ')

def reverse_Count(number, seconds):
    if False:
        i = 10
        return i + 15
    cont = number
    if number < 0:
        return print('The number must be positive')
    for x in range(number + 1):
        print(cont)
        sleep(seconds)
        cont = cont - 1
reverse_Count(number_to_count, seconds_to_wait)