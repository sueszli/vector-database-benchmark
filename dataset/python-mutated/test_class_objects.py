"""Class Definition Syntax.

@see: https://docs.python.org/3/tutorial/classes.html#class-objects

After defining the class attributes to a class, the class object can be created by assigning the
object to a variable. The created object would have instance attributes associated with it.
"""

def test_class_objects():
    if False:
        i = 10
        return i + 15
    'Class Objects.\n\n    Class objects support two kinds of operations:\n    - attribute references\n    - instantiation.\n    '

    class ComplexNumber:
        """Example of the complex numbers class"""
        real = 0
        imaginary = 0

        def get_real(self):
            if False:
                i = 10
                return i + 15
            'Return real part of complex number.'
            return self.real

        def get_imaginary(self):
            if False:
                for i in range(10):
                    print('nop')
            'Return imaginary part of complex number.'
            return self.imaginary
    assert ComplexNumber.real == 0
    assert ComplexNumber.__doc__ == 'Example of the complex numbers class'
    ComplexNumber.real = 10
    assert ComplexNumber.real == 10
    complex_number = ComplexNumber()
    assert complex_number.real == 10
    assert complex_number.get_real() == 10
    ComplexNumber.real = 10
    assert ComplexNumber.real == 10

    class ComplexNumberWithConstructor:
        """Example of the class with constructor"""

        def __init__(self, real_part, imaginary_part):
            if False:
                return 10
            self.real = real_part
            self.imaginary = imaginary_part

        def get_real(self):
            if False:
                return 10
            'Return real part of complex number.'
            return self.real

        def get_imaginary(self):
            if False:
                i = 10
                return i + 15
            'Return imaginary part of complex number.'
            return self.imaginary
    complex_number = ComplexNumberWithConstructor(3.0, -4.5)
    assert complex_number.real, complex_number.imaginary == (3.0, -4.5)