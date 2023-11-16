def fizz_buzz():
    if False:
        i = 10
        return i + 15
    for i in range(1, 21):
        if i % 3 == 0:
            print(i, 'fizz')
        elif i % 5 == 0:
            print(i, 'buzz')
        elif i % 3 == 0 and i % 5 == 0:
            print(i, 'fizzbuzz')
fizz_buzz()