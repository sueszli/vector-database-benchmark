def bizz_buzz(range_value: int, reference_one: int, reference_two: int):
    if False:
        return 10
    for i in range(0, range_value + 1):
        if i % reference_one == 0 and i % reference_two == 0:
            print('fizzbuzz')
        elif i % reference_one == 0:
            print('fizz')
        elif i % reference_two == 0:
            print('buzz')
        else:
            print(i)
bizz_buzz(100, 3, 5)