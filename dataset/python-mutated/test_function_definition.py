"""Function Definition

@see: https://docs.python.org/3/tutorial/controlflow.html#defining-functions
@see: https://www.thecodeship.com/patterns/guide-to-python-function-decorators/

The keyword def introduces a function definition. It must be followed by the function name and the
parenthesized list of formal parameters. The statements that form the body of the function start at
the next line, and must be indented.
"""

def fibonacci_function_example(number_limit):
    if False:
        while True:
            i = 10
    'Generate a Fibonacci series up to number_limit.\n\n    The first statement of the function body can optionally be a string literal; this string\n    literal is the function’s documentation string, or docstring. There are tools which use\n    docstrings to automatically produce online or printed documentation, or to let the user\n    interactively browse through code; it’s good practice to include docstrings in code that you\n    write, so make a habit of it.\n    '
    fibonacci_list = []
    (previous_number, current_number) = (0, 1)
    while previous_number < number_limit:
        fibonacci_list.append(previous_number)
        (previous_number, current_number) = (current_number, previous_number + current_number)
    return fibonacci_list

def test_function_definition():
    if False:
        print('Hello World!')
    'Function Definition'
    assert fibonacci_function_example(300) == [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233]
    fibonacci_function_clone = fibonacci_function_example
    assert fibonacci_function_clone(300) == [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233]

    def greet(name):
        if False:
            for i in range(10):
                print('nop')
        return 'Hello, ' + name
    greet_someone = greet
    assert greet_someone('John') == 'Hello, John'

    def greet_again(name):
        if False:
            print('Hello World!')

        def get_message():
            if False:
                i = 10
                return i + 15
            return 'Hello, '
        result = get_message() + name
        return result
    assert greet_again('John') == 'Hello, John'

    def greet_one_more(name):
        if False:
            while True:
                i = 10
        return 'Hello, ' + name

    def call_func(func):
        if False:
            print('Hello World!')
        other_name = 'John'
        return func(other_name)
    assert call_func(greet_one_more) == 'Hello, John'

    def compose_greet_func():
        if False:
            while True:
                i = 10

        def get_message():
            if False:
                return 10
            return 'Hello there!'
        return get_message
    greet_function = compose_greet_func()
    assert greet_function() == 'Hello there!'

    def compose_greet_func_with_closure(name):
        if False:
            return 10

        def get_message():
            if False:
                return 10
            return 'Hello there, ' + name + '!'
        return get_message
    greet_with_closure = compose_greet_func_with_closure('John')
    assert greet_with_closure() == 'Hello there, John!'