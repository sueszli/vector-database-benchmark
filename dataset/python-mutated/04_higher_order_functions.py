from functools import reduce

def sum_one(value):
    if False:
        print('Hello World!')
    return value + 1

def sum_five(value):
    if False:
        i = 10
        return i + 15
    return value + 5

def sum_two_values_and_add_value(first_value, second_value, f_sum):
    if False:
        for i in range(10):
            print('nop')
    return f_sum(first_value + second_value)
print(sum_two_values_and_add_value(5, 2, sum_one))
print(sum_two_values_and_add_value(5, 2, sum_five))

def sum_ten(original_value):
    if False:
        i = 10
        return i + 15

    def add(value):
        if False:
            return 10
        return value + 10 + original_value
    return add
add_closure = sum_ten(1)
print(add_closure(5))
print(sum_ten(5)(1))
numbers = [2, 5, 10, 21, 3, 30]

def multiply_two(number):
    if False:
        for i in range(10):
            print('nop')
    return number * 2
print(list(map(multiply_two, numbers)))
print(list(map(lambda number: number * 2, numbers)))

def filter_greater_than_ten(number):
    if False:
        print('Hello World!')
    if number > 10:
        return True
    return False
print(list(filter(filter_greater_than_ten, numbers)))
print(list(filter(lambda number: number > 10, numbers)))

def sum_two_values(first_value, second_value):
    if False:
        i = 10
        return i + 15
    return first_value + second_value
print(reduce(sum_two_values, numbers))