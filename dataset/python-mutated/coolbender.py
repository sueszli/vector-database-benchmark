def thechecker(n, m):
    if False:
        print('Hello World!')
    return True if n % m == 0 else False

def thefizzbuzz():
    if False:
        print('Hello World!')
    for num in range(1, 101, 1):
        fizz = thechecker(num, 3)
        buzz = thechecker(num, 5)
        if fizz and buzz:
            print('fizzbuzz \n')
        elif fizz:
            print('fizz \n')
        elif buzz:
            print('buzz \n')
        else:
            print(str(num) + '\n')
thefizzbuzz()