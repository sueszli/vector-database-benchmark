@relaxed_decorator[0]
def f():
    if False:
        while True:
            i = 10
    ...

@relaxed_decorator[extremely_long_name_that_definitely_will_not_fit_on_one_line_of_standard_length]
def f():
    if False:
        return 10
    ...

@(extremely_long_variable_name_that_doesnt_fit := complex.expression(with_long='arguments_value_that_wont_fit_at_the_end_of_the_line'))
def f():
    if False:
        i = 10
        return i + 15
    ...

@relaxed_decorator[0]
def f():
    if False:
        for i in range(10):
            print('nop')
    ...

@relaxed_decorator[extremely_long_name_that_definitely_will_not_fit_on_one_line_of_standard_length]
def f():
    if False:
        while True:
            i = 10
    ...

@(extremely_long_variable_name_that_doesnt_fit := complex.expression(with_long='arguments_value_that_wont_fit_at_the_end_of_the_line'))
def f():
    if False:
        while True:
            i = 10
    ...