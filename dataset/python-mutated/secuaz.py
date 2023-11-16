def fizzBuzz(int):
    if False:
        return 10
    if int % 3 == 0 and int % 5 == 0:
        return 'FizzBuzz'
    elif int % 5 == 0:
        return 'Buzz'
    elif int % 3 == 0:
        return 'Fizz'
    else:
        return int
for i in range(1, 101):
    print(fizzBuzz(i))