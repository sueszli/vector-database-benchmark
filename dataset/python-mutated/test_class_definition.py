"""Class Definition Syntax.

@see: https://docs.python.org/3/tutorial/classes.html

Python is an object oriented programming language.
Almost everything in Python is an object, with its properties and methods.
A Class is like an object constructor, or a "blueprint" for creating objects.
"""

def test_class_definition():
    if False:
        while True:
            i = 10
    'Class definition.'

    class GreetingClass:
        """Example of the class definition

        This class contains two public methods and doesn't contain constructor.
        """
        name = 'user'

        def say_hello(self):
            if False:
                print('Hello World!')
            'Class method.'
            return 'Hello ' + self.name

        def say_goodbye(self):
            if False:
                while True:
                    i = 10
            'Class method.'
            return 'Goodbye ' + self.name
    greeter = GreetingClass()
    assert greeter.say_hello() == 'Hello user'
    assert greeter.say_goodbye() == 'Goodbye user'