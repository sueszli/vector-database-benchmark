"""Class Definition Syntax.

@see: https://docs.python.org/3/tutorial/classes.html#method-objects

Classes can have two types of attribute references: data or methods. Class methods are called
by [variable_name].[method_name]([parameters]) as opposed to class data which lacks the ().
"""

class MyCounter:
    """A simple example of the counter class"""
    counter = 10

    def get_counter(self):
        if False:
            while True:
                i = 10
        'Return the counter'
        return self.counter

    def increment_counter(self):
        if False:
            return 10
        'Increment the counter'
        self.counter += 1
        return self.counter

def test_method_objects():
    if False:
        i = 10
        return i + 15
    'Method Objects.'
    counter = MyCounter()
    assert counter.get_counter() == 10
    get_counter = counter.get_counter
    assert get_counter() == 10
    assert counter.get_counter() == 10
    assert MyCounter.get_counter(counter) == 10