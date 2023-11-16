"""Class and Instance Variables.

@see: https://docs.python.org/3/tutorial/classes.html#class-and-instance-variables

Generally speaking, instance variables are for data unique to each instance and class variables are
for attributes and methods shared by all instances of the class.
"""

def test_class_and_instance_variables():
    if False:
        return 10
    'Class and Instance Variables.'

    class Dog:
        """Dog class example"""
        kind = 'canine'

        def __init__(self, name):
            if False:
                print('Hello World!')
            self.name = name
    fido = Dog('Fido')
    buddy = Dog('Buddy')
    assert fido.kind == 'canine'
    assert buddy.kind == 'canine'
    assert fido.name == 'Fido'
    assert buddy.name == 'Buddy'

    class DogWithSharedTricks:
        """Dog class example with wrong shared variable usage"""
        tricks = []

        def __init__(self, name):
            if False:
                return 10
            self.name = name

        def add_trick(self, trick):
            if False:
                i = 10
                return i + 15
            'Add trick to the dog\n\n            This function illustrate mistaken use of mutable class variable tricks (see below).\n            '
            self.tricks.append(trick)
    fido = DogWithSharedTricks('Fido')
    buddy = DogWithSharedTricks('Buddy')
    fido.add_trick('roll over')
    buddy.add_trick('play dead')
    assert fido.tricks == ['roll over', 'play dead']
    assert buddy.tricks == ['roll over', 'play dead']

    class DogWithTricks:
        """Dog class example"""

        def __init__(self, name):
            if False:
                while True:
                    i = 10
            self.name = name
            self.tricks = []

        def add_trick(self, trick):
            if False:
                i = 10
                return i + 15
            'Add trick to the dog\n\n            This function illustrate a correct use of mutable class variable tricks (see below).\n            '
            self.tricks.append(trick)
    fido = DogWithTricks('Fido')
    buddy = DogWithTricks('Buddy')
    fido.add_trick('roll over')
    buddy.add_trick('play dead')
    assert fido.tricks == ['roll over']
    assert buddy.tricks == ['play dead']