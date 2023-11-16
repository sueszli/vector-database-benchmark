def fizzbuzz(number) -> int:
    if False:
        for i in range(10):
            print('nop')
    for number in range(1, 101):
        if number % 3 == 0:
            print('fizz')
        elif number % 5 == 0:
            print('buzz')
        elif number % 3 == 0 and number % 5 == 0:
            print('fizzbuzz')
        else:
            print(number)
fizzbuzz(number=range)