import numpy as np
import NumCppPy as NumCpp

def test_uint32_bits():
    if False:
        for i in range(10):
            print('nop')
    assert NumCpp.DtypeIntoUint32.bits() == 32

def test_uint32_epsilon():
    if False:
        print('Hello World!')
    assert NumCpp.DtypeIntoUint32.epsilon() == 0

def test_uint32_isInteger():
    if False:
        i = 10
        return i + 15
    assert NumCpp.DtypeIntoUint32.isInteger()

def test_uint32_isSigned():
    if False:
        return 10
    assert not NumCpp.DtypeIntoUint32.isSigned()

def test_uint32_max():
    if False:
        i = 10
        return i + 15
    assert NumCpp.DtypeIntoUint32.max() == np.iinfo(np.uint32).max

def test_uint32_min():
    if False:
        i = 10
        return i + 15
    assert NumCpp.DtypeIntoUint32.min() == np.iinfo(np.uint32).min

def test_complex_bits():
    if False:
        return 10
    assert NumCpp.DtypeInfoComplexDouble.bits()

def test_complex_epsilon():
    if False:
        return 10
    assert NumCpp.DtypeInfoComplexDouble.epsilon()

def test_complex_isInteger():
    if False:
        return 10
    assert not NumCpp.DtypeInfoComplexDouble.isInteger()

def test_complex_isSigned():
    if False:
        while True:
            i = 10
    assert NumCpp.DtypeInfoComplexDouble.isSigned()

def test_complex_max():
    if False:
        i = 10
        return i + 15
    assert NumCpp.DtypeInfoComplexDouble.max()

def test_complex_min():
    if False:
        print('Hello World!')
    assert NumCpp.DtypeInfoComplexDouble.min()