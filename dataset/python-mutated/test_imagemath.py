import pytest
from PIL import Image, ImageMath

def pixel(im):
    if False:
        while True:
            i = 10
    if hasattr(im, 'im'):
        return f'{im.mode} {repr(im.getpixel((0, 0)))}'
    if isinstance(im, int):
        return int(im)
A = Image.new('L', (1, 1), 1)
B = Image.new('L', (1, 1), 2)
Z = Image.new('L', (1, 1), 0)
F = Image.new('F', (1, 1), 3)
I = Image.new('I', (1, 1), 4)
A2 = A.resize((2, 2))
B2 = B.resize((2, 2))
images = {'A': A, 'B': B, 'F': F, 'I': I}

def test_sanity():
    if False:
        for i in range(10):
            print('nop')
    assert ImageMath.eval('1') == 1
    assert ImageMath.eval('1+A', A=2) == 3
    assert pixel(ImageMath.eval('A+B', A=A, B=B)) == 'I 3'
    assert pixel(ImageMath.eval('A+B', images)) == 'I 3'
    assert pixel(ImageMath.eval('float(A)+B', images)) == 'F 3.0'
    assert pixel(ImageMath.eval('int(float(A)+B)', images)) == 'I 3'

def test_ops():
    if False:
        while True:
            i = 10
    assert pixel(ImageMath.eval('-A', images)) == 'I -1'
    assert pixel(ImageMath.eval('+B', images)) == 'L 2'
    assert pixel(ImageMath.eval('A+B', images)) == 'I 3'
    assert pixel(ImageMath.eval('A-B', images)) == 'I -1'
    assert pixel(ImageMath.eval('A*B', images)) == 'I 2'
    assert pixel(ImageMath.eval('A/B', images)) == 'I 0'
    assert pixel(ImageMath.eval('B**2', images)) == 'I 4'
    assert pixel(ImageMath.eval('B**33', images)) == 'I 2147483647'
    assert pixel(ImageMath.eval('float(A)+B', images)) == 'F 3.0'
    assert pixel(ImageMath.eval('float(A)-B', images)) == 'F -1.0'
    assert pixel(ImageMath.eval('float(A)*B', images)) == 'F 2.0'
    assert pixel(ImageMath.eval('float(A)/B', images)) == 'F 0.5'
    assert pixel(ImageMath.eval('float(B)**2', images)) == 'F 4.0'
    assert pixel(ImageMath.eval('float(B)**33', images)) == 'F 8589934592.0'

@pytest.mark.parametrize('expression', ("exec('pass')", "(lambda: exec('pass'))()", "(lambda: (lambda: exec('pass'))())()"))
def test_prevent_exec(expression):
    if False:
        return 10
    with pytest.raises(ValueError):
        ImageMath.eval(expression)

def test_logical():
    if False:
        print('Hello World!')
    assert pixel(ImageMath.eval('not A', images)) == 0
    assert pixel(ImageMath.eval('A and B', images)) == 'L 2'
    assert pixel(ImageMath.eval('A or B', images)) == 'L 1'

def test_convert():
    if False:
        while True:
            i = 10
    assert pixel(ImageMath.eval("convert(A+B, 'L')", images)) == 'L 3'
    assert pixel(ImageMath.eval("convert(A+B, '1')", images)) == '1 0'
    assert pixel(ImageMath.eval("convert(A+B, 'RGB')", images)) == 'RGB (3, 3, 3)'

def test_compare():
    if False:
        print('Hello World!')
    assert pixel(ImageMath.eval('min(A, B)', images)) == 'I 1'
    assert pixel(ImageMath.eval('max(A, B)', images)) == 'I 2'
    assert pixel(ImageMath.eval('A == 1', images)) == 'I 1'
    assert pixel(ImageMath.eval('A == 2', images)) == 'I 0'

def test_one_image_larger():
    if False:
        i = 10
        return i + 15
    assert pixel(ImageMath.eval('A+B', A=A2, B=B)) == 'I 3'
    assert pixel(ImageMath.eval('A+B', A=A, B=B2)) == 'I 3'

def test_abs():
    if False:
        print('Hello World!')
    assert pixel(ImageMath.eval('abs(A)', A=A)) == 'I 1'
    assert pixel(ImageMath.eval('abs(B)', B=B)) == 'I 2'

def test_binary_mod():
    if False:
        for i in range(10):
            print('nop')
    assert pixel(ImageMath.eval('A%A', A=A)) == 'I 0'
    assert pixel(ImageMath.eval('B%B', B=B)) == 'I 0'
    assert pixel(ImageMath.eval('A%B', A=A, B=B)) == 'I 1'
    assert pixel(ImageMath.eval('B%A', A=A, B=B)) == 'I 0'
    assert pixel(ImageMath.eval('Z%A', A=A, Z=Z)) == 'I 0'
    assert pixel(ImageMath.eval('Z%B', B=B, Z=Z)) == 'I 0'

def test_bitwise_invert():
    if False:
        while True:
            i = 10
    assert pixel(ImageMath.eval('~Z', Z=Z)) == 'I -1'
    assert pixel(ImageMath.eval('~A', A=A)) == 'I -2'
    assert pixel(ImageMath.eval('~B', B=B)) == 'I -3'

def test_bitwise_and():
    if False:
        return 10
    assert pixel(ImageMath.eval('Z&Z', A=A, Z=Z)) == 'I 0'
    assert pixel(ImageMath.eval('Z&A', A=A, Z=Z)) == 'I 0'
    assert pixel(ImageMath.eval('A&Z', A=A, Z=Z)) == 'I 0'
    assert pixel(ImageMath.eval('A&A', A=A, Z=Z)) == 'I 1'

def test_bitwise_or():
    if False:
        return 10
    assert pixel(ImageMath.eval('Z|Z', A=A, Z=Z)) == 'I 0'
    assert pixel(ImageMath.eval('Z|A', A=A, Z=Z)) == 'I 1'
    assert pixel(ImageMath.eval('A|Z', A=A, Z=Z)) == 'I 1'
    assert pixel(ImageMath.eval('A|A', A=A, Z=Z)) == 'I 1'

def test_bitwise_xor():
    if False:
        for i in range(10):
            print('nop')
    assert pixel(ImageMath.eval('Z^Z', A=A, Z=Z)) == 'I 0'
    assert pixel(ImageMath.eval('Z^A', A=A, Z=Z)) == 'I 1'
    assert pixel(ImageMath.eval('A^Z', A=A, Z=Z)) == 'I 1'
    assert pixel(ImageMath.eval('A^A', A=A, Z=Z)) == 'I 0'

def test_bitwise_leftshift():
    if False:
        return 10
    assert pixel(ImageMath.eval('Z<<0', Z=Z)) == 'I 0'
    assert pixel(ImageMath.eval('Z<<1', Z=Z)) == 'I 0'
    assert pixel(ImageMath.eval('A<<0', A=A)) == 'I 1'
    assert pixel(ImageMath.eval('A<<1', A=A)) == 'I 2'

def test_bitwise_rightshift():
    if False:
        i = 10
        return i + 15
    assert pixel(ImageMath.eval('Z>>0', Z=Z)) == 'I 0'
    assert pixel(ImageMath.eval('Z>>1', Z=Z)) == 'I 0'
    assert pixel(ImageMath.eval('A>>0', A=A)) == 'I 1'
    assert pixel(ImageMath.eval('A>>1', A=A)) == 'I 0'

def test_logical_eq():
    if False:
        print('Hello World!')
    assert pixel(ImageMath.eval('A==A', A=A)) == 'I 1'
    assert pixel(ImageMath.eval('B==B', B=B)) == 'I 1'
    assert pixel(ImageMath.eval('A==B', A=A, B=B)) == 'I 0'
    assert pixel(ImageMath.eval('B==A', A=A, B=B)) == 'I 0'

def test_logical_ne():
    if False:
        for i in range(10):
            print('nop')
    assert pixel(ImageMath.eval('A!=A', A=A)) == 'I 0'
    assert pixel(ImageMath.eval('B!=B', B=B)) == 'I 0'
    assert pixel(ImageMath.eval('A!=B', A=A, B=B)) == 'I 1'
    assert pixel(ImageMath.eval('B!=A', A=A, B=B)) == 'I 1'

def test_logical_lt():
    if False:
        for i in range(10):
            print('nop')
    assert pixel(ImageMath.eval('A<A', A=A)) == 'I 0'
    assert pixel(ImageMath.eval('B<B', B=B)) == 'I 0'
    assert pixel(ImageMath.eval('A<B', A=A, B=B)) == 'I 1'
    assert pixel(ImageMath.eval('B<A', A=A, B=B)) == 'I 0'

def test_logical_le():
    if False:
        return 10
    assert pixel(ImageMath.eval('A<=A', A=A)) == 'I 1'
    assert pixel(ImageMath.eval('B<=B', B=B)) == 'I 1'
    assert pixel(ImageMath.eval('A<=B', A=A, B=B)) == 'I 1'
    assert pixel(ImageMath.eval('B<=A', A=A, B=B)) == 'I 0'

def test_logical_gt():
    if False:
        print('Hello World!')
    assert pixel(ImageMath.eval('A>A', A=A)) == 'I 0'
    assert pixel(ImageMath.eval('B>B', B=B)) == 'I 0'
    assert pixel(ImageMath.eval('A>B', A=A, B=B)) == 'I 0'
    assert pixel(ImageMath.eval('B>A', A=A, B=B)) == 'I 1'

def test_logical_ge():
    if False:
        print('Hello World!')
    assert pixel(ImageMath.eval('A>=A', A=A)) == 'I 1'
    assert pixel(ImageMath.eval('B>=B', B=B)) == 'I 1'
    assert pixel(ImageMath.eval('A>=B', A=A, B=B)) == 'I 0'
    assert pixel(ImageMath.eval('B>=A', A=A, B=B)) == 'I 1'

def test_logical_equal():
    if False:
        return 10
    assert pixel(ImageMath.eval('equal(A, A)', A=A)) == 'I 1'
    assert pixel(ImageMath.eval('equal(B, B)', B=B)) == 'I 1'
    assert pixel(ImageMath.eval('equal(Z, Z)', Z=Z)) == 'I 1'
    assert pixel(ImageMath.eval('equal(A, B)', A=A, B=B)) == 'I 0'
    assert pixel(ImageMath.eval('equal(B, A)', A=A, B=B)) == 'I 0'
    assert pixel(ImageMath.eval('equal(A, Z)', A=A, Z=Z)) == 'I 0'

def test_logical_not_equal():
    if False:
        for i in range(10):
            print('nop')
    assert pixel(ImageMath.eval('notequal(A, A)', A=A)) == 'I 0'
    assert pixel(ImageMath.eval('notequal(B, B)', B=B)) == 'I 0'
    assert pixel(ImageMath.eval('notequal(Z, Z)', Z=Z)) == 'I 0'
    assert pixel(ImageMath.eval('notequal(A, B)', A=A, B=B)) == 'I 1'
    assert pixel(ImageMath.eval('notequal(B, A)', A=A, B=B)) == 'I 1'
    assert pixel(ImageMath.eval('notequal(A, Z)', A=A, Z=Z)) == 'I 1'