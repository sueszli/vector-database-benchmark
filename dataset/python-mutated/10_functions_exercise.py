def print_pattern(n=5):
    if False:
        while True:
            i = 10
    "\n    :param n: Integer number representing number of lines\n    to be printed in a pattern. If n=3 it will print,\n      *\n      **\n      ***\n    If n=4, it will print,\n      *\n      **\n      ***\n      ****\n    Default value for n is 5. So if function caller doesn't\n    supply the input number then it will assume it to be 5\n    :return: None\n    "
    for i in range(n):
        s = ''
        for j in range(i + 1):
            s = s + '*'
        print(s)

def calculate_area(dimension1, dimension2, shape='triangle'):
    if False:
        i = 10
        return i + 15
    '\n    :param dimension1: In case of triangle it is "base". For rectangle it is "length".\n    :param dimension2: In case of triangle it is "height". For rectangle it is "width".\n    :param shape: Either "triangle" or "rectangle"\n    :return: Area of a shape\n    '
    if shape == 'triangle':
        area = 1 / 2 * (dimension1 * dimension2)
    elif shape == 'rectangle':
        area = dimension1 * dimension2
    else:
        print('Error: Input shape is neither triangle nor rectangle.')
        area = None
    return area
base = 10
height = 5
triangle_area = calculate_area(base, height, 'triangle')
print('Area of triangle is:', triangle_area)
length = 20
width = 30
rectangle_area = calculate_area(length, width, 'rectangle')
print('Area of rectangle is:', rectangle_area)
triangle_area = calculate_area(base, height)
print('Area of triangle with no shape supplied: ', triangle_area)
print('Print pattern with input=3')
print_pattern(3)
print('Print pattern with input=4')
print_pattern(4)
print('Print pattern with no input number')
print_pattern()