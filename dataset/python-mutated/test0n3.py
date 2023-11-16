import datetime

def is_friday_13(month, year):
    if False:
        while True:
            i = 10
    date = datetime.date(year, month, 13)
    if date.weekday() == 4:
        return True
    else:
        return False

def test_is_friday_13():
    if False:
        print('Hello World!')
    tests = {'input': [[4, 2022], [8, 2024], [5, 2022], [9, 2024]], 'output': [False, False, True, True]}
    errors = 0
    for (index, test) in enumerate(tests['input']):
        resp = is_friday_13(test[0], test[1])
        expected = tests['output'][index]
        if resp != expected:
            errors += 1
            print(f'\n\noriginal: {test}')
            print(resp)
            print(f'expected: {expected}')
    print(f"\nTests{(' not ' if errors != 0 else ' ')}passed, {errors} errors\n")
test_is_friday_13()