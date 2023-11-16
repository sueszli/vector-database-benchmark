"""Function Decorators.

@see: https://www.thecodeship.com/patterns/guide-to-python-function-decorators/

Function decorators are simply wrappers to existing functions. In the context of design patterns,
decorators dynamically alter the functionality of a function, method or class without having to
directly use subclasses. This is ideal when you need to extend the functionality of functions that
you don't want to modify. We can implement the decorator pattern anywhere, but Python facilitates
the implementation by providing much more expressive features and syntax for that.
"""

def test_function_decorators():
    if False:
        return 10
    'Function Decorators.'

    def greeting(name):
        if False:
            return 10
        return 'Hello, {0}!'.format(name)

    def decorate_with_p(func):
        if False:
            print('Hello World!')

        def function_wrapper(name):
            if False:
                i = 10
                return i + 15
            return '<p>{0}</p>'.format(func(name))
        return function_wrapper
    my_get_text = decorate_with_p(greeting)
    assert my_get_text('John') == '<p>Hello, John!</p>'
    assert greeting('John') == 'Hello, John!'

    @decorate_with_p
    def greeting_with_p(name):
        if False:
            for i in range(10):
                print('nop')
        return 'Hello, {0}!'.format(name)
    assert greeting_with_p('John') == '<p>Hello, John!</p>'

    def decorate_with_div(func):
        if False:
            for i in range(10):
                print('nop')

        def function_wrapper(text):
            if False:
                return 10
            return '<div>{0}</div>'.format(func(text))
        return function_wrapper

    @decorate_with_div
    @decorate_with_p
    def greeting_with_div_p(name):
        if False:
            return 10
        return 'Hello, {0}!'.format(name)
    assert greeting_with_div_p('John') == '<div><p>Hello, John!</p></div>'

    def tags(tag_name):
        if False:
            for i in range(10):
                print('nop')

        def tags_decorator(func):
            if False:
                print('Hello World!')

            def func_wrapper(name):
                if False:
                    for i in range(10):
                        print('nop')
                return '<{0}>{1}</{0}>'.format(tag_name, func(name))
            return func_wrapper
        return tags_decorator

    @tags('div')
    @tags('p')
    def greeting_with_tags(name):
        if False:
            return 10
        return 'Hello, {0}!'.format(name)
    assert greeting_with_tags('John') == '<div><p>Hello, John!</p></div>'