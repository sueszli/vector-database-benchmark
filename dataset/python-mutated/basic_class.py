"""
A class is made up of methods and state. This allows code and data to be
combined as one logical entity. This module defines a basic car class,
creates a car instance and uses it for demonstration purposes.
"""
from inspect import isfunction, ismethod, signature

class Car:
    """Basic definition of a car.

    We begin with a simple mental model of what a car is. That way, we
    can start exploring the core concepts that are associated with a
    class definition.
    """

    def __init__(self, make, model, year, miles):
        if False:
            i = 10
            return i + 15
        'Constructor logic.'
        self.make = make
        self.model = model
        self.year = year
        self.miles = miles

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        'Formal representation for developers.'
        return f'<Car make={self.make} model={self.model} year={self.year}>'

    def __str__(self):
        if False:
            while True:
                i = 10
        'Informal representation for users.'
        return f'{self.make} {self.model} ({self.year})'

    def drive(self, rate_in_mph):
        if False:
            return 10
        'Drive car at a certain rate in MPH.'
        return f'{self} is driving at {rate_in_mph} MPH'

def main():
    if False:
        return 10
    car = Car('Bumble', 'Bee', 2000, 200000.0)
    assert repr(car) == '<Car make=Bumble model=Bee year=2000>'
    assert str(car) == 'Bumble Bee (2000)'
    assert car.drive(75) == 'Bumble Bee (2000) is driving at 75 MPH'
    assert issubclass(Car, object) and isinstance(Car, object)
    driving = getattr(car, 'drive')
    assert driving == car.drive
    assert driving.__self__ == car
    assert ismethod(driving) and (not isfunction(driving))
    driving_params = signature(driving).parameters
    assert len(driving_params) == 1
    assert 'rate_in_mph' in driving_params
if __name__ == '__main__':
    main()