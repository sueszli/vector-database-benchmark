"""Assignment operators

@see: https://www.w3schools.com/python/python_operators.asp

Assignment operators are used to assign values to variables
"""

def test_assignment_operator():
    if False:
        return 10
    'Assignment operator '
    number = 5
    assert number == 5
    (first_variable, second_variable) = (0, 1)
    assert first_variable == 0
    assert second_variable == 1
    (first_variable, second_variable) = (second_variable, first_variable)
    assert first_variable == 1
    assert second_variable == 0

def test_augmented_assignment_operators():
    if False:
        i = 10
        return i + 15
    'Assignment operator combined with arithmetic and bitwise operators'
    number = 5
    number += 3
    assert number == 8
    number = 5
    number -= 3
    assert number == 2
    number = 5
    number *= 3
    assert number == 15
    number = 8
    number /= 4
    assert number == 2
    number = 8
    number %= 3
    assert number == 2
    number = 5
    number %= 3
    assert number == 2
    number = 5
    number //= 3
    assert number == 1
    number = 5
    number **= 3
    assert number == 125
    number = 5
    number &= 3
    assert number == 1
    number = 5
    number |= 3
    assert number == 7
    number = 5
    number ^= 3
    assert number == 6
    number = 5
    number >>= 3
    assert number == 0
    number = 5
    number <<= 3
    assert number == 40