from datetime import date

def friday13(year, month):
    if False:
        i = 10
        return i + 15
    return date(year, month, 13).weekday() == 4
if __name__ == '__main__':
    test_values = [True, False, False, False, False, False, False, False, False, True, False, False]
    for i in range(1, 13):
        assert friday13(2023, i) == test_values[i - 1], f'Month {i} should be {test_values[i - 1]}'