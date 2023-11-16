def f(a):
    if False:
        i = 10
        return i + 15
    d = {'key': 'value'}
    tup = (1,)

def f2(a, b):
    if False:
        return 10
    d = {'key': 'value', 'key2': 'value2'}
    tup = (1, 2)

def f(a: int=1):
    if False:
        return 10
    call(arg={'explode': 'this'})
    call2(arg=[1, 2, 3])
    x = {'a': 1, 'b': 2}['a']
    if a == {'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5, 'f': 6, 'g': 7, 'h': 8}['a']:
        pass

def xxxxxxxxxxxxxxxxxxxxxxxxxxxx() -> Set['xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx']:
    if False:
        while True:
            i = 10
    json = {'k': {'k2': {'k3': [1]}}}

def some_function_with_a_really_long_name() -> returning_a_deeply_nested_import_of_a_type_i_suppose:
    if False:
        for i in range(10):
            print('nop')
    pass

def some_method_with_a_really_long_name(very_long_parameter_so_yeah: str, another_long_parameter: int) -> another_case_of_returning_a_deeply_nested_import_of_a_type_i_suppose_cause_why_not:
    if False:
        i = 10
        return i + 15
    pass

def func() -> also_super_long_type_annotation_that_may_cause_an_AST_related_crash_in_black(this_shouldn_t_get_a_trailing_comma_too):
    if False:
        print('Hello World!')
    pass

def func() -> also_super_long_type_annotation_that_may_cause_an_AST_related_crash_in_black(this_shouldn_t_get_a_trailing_comma_too):
    if False:
        print('Hello World!')
    pass
some_module.some_function(argument1, (one_element_tuple,), argument4, argument5, argument6)
some_module.some_function(argument1, (one, two), argument4, argument5, argument6)

def f(a):
    if False:
        for i in range(10):
            print('nop')
    d = {'key': 'value'}
    tup = (1,)

def f2(a, b):
    if False:
        return 10
    d = {'key': 'value', 'key2': 'value2'}
    tup = (1, 2)

def f(a: int=1):
    if False:
        while True:
            i = 10
    call(arg={'explode': 'this'})
    call2(arg=[1, 2, 3])
    x = {'a': 1, 'b': 2}['a']
    if a == {'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5, 'f': 6, 'g': 7, 'h': 8}['a']:
        pass

def xxxxxxxxxxxxxxxxxxxxxxxxxxxx() -> Set['xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx']:
    if False:
        i = 10
        return i + 15
    json = {'k': {'k2': {'k3': [1]}}}

def some_function_with_a_really_long_name() -> returning_a_deeply_nested_import_of_a_type_i_suppose:
    if False:
        i = 10
        return i + 15
    pass

def some_method_with_a_really_long_name(very_long_parameter_so_yeah: str, another_long_parameter: int) -> another_case_of_returning_a_deeply_nested_import_of_a_type_i_suppose_cause_why_not:
    if False:
        while True:
            i = 10
    pass

def func() -> also_super_long_type_annotation_that_may_cause_an_AST_related_crash_in_black(this_shouldn_t_get_a_trailing_comma_too):
    if False:
        for i in range(10):
            print('nop')
    pass

def func() -> also_super_long_type_annotation_that_may_cause_an_AST_related_crash_in_black(this_shouldn_t_get_a_trailing_comma_too):
    if False:
        for i in range(10):
            print('nop')
    pass
some_module.some_function(argument1, (one_element_tuple,), argument4, argument5, argument6)
some_module.some_function(argument1, (one, two), argument4, argument5, argument6)