ONE_DAY = 60 * 60 * 60 * 24
MANY_DAYS = 5 * ONE_DAY
ONE = 1
TWO = 2 * ONE
FIVE = 5 * ONE

def bad(var):
    if False:
        for i in range(10):
            print('nop')
    print(var)
bad(var=2)
bad(var=TWO)
bad(var=3)
bad(var=FIVE)
bad(var=ONE_DAY)
bad(var=3000)
bad(var=MANY_DAYS)