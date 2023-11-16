def my_function():
    if False:
        return 10
    print('Esto es una función')
my_function()
my_function()
my_function()

def sum_two_values(first_value: int, second_value):
    if False:
        return 10
    print(first_value + second_value)
sum_two_values(5, 7)
sum_two_values(54754, 71231)
sum_two_values('5', '7')
sum_two_values(1.4, 5.2)

def sum_two_values_with_return(first_value, second_value):
    if False:
        for i in range(10):
            print('nop')
    my_sum = first_value + second_value
    return my_sum
my_result = sum_two_values(1.4, 5.2)
print(my_result)
my_result = sum_two_values_with_return(10, 5)
print(my_result)

def print_name(name, surname):
    if False:
        return 10
    print(f'{name} {surname}')
print_name(surname='Moure', name='Brais')

def print_name_with_default(name, surname, alias='Sin alias'):
    if False:
        for i in range(10):
            print('nop')
    print(f'{name} {surname} {alias}')
print_name_with_default('Brais', 'Moure')
print_name_with_default('Brais', 'Moure', 'MoureDev')

def print_upper_texts(*texts):
    if False:
        return 10
    print(type(texts))
    for text in texts:
        print(text.upper())
print_upper_texts('Hola', 'Python', 'MoureDev')
print_upper_texts('Hola')