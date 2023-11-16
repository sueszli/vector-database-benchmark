def functionNoParameters():
    if False:
        i = 10
        return i + 15
    pass
print('Call a function with no parameters with a plain argument:')
try:
    functionNoParameters(1)
except TypeError as e:
    print(repr(e))
print('Call a function with no parameters with a keyword argument:')
try:
    functionNoParameters(z=1)
except TypeError as e:
    print(repr(e))

def functionOneParameter(a):
    if False:
        print('Hello World!')
    print(a)
print('Call a function with one parameter with two plain arguments:')
try:
    functionOneParameter(1, 1)
except TypeError as e:
    print(repr(e))
print('Call a function with one parameter too many, and duplicate arguments:')
try:
    functionOneParameter(6, *(1, 2, 3), a=4)
except TypeError as e:
    print(repr(e))
print('Call a function with two parameters with three plain arguments:')

def functionTwoParameters(a, b):
    if False:
        return 10
    print(a, b)
try:
    functionTwoParameters(1, 2, 3)
except TypeError as e:
    print(repr(e))
print('Call a function with two parameters with one plain argument:')
try:
    functionTwoParameters(1)
except TypeError as e:
    print(repr(e))
print('Call a function with two parameters with three plain arguments:')
try:
    functionTwoParameters(1, 2, 3)
except TypeError as e:
    print(repr(e))
print('Call a function with two parameters with one keyword argument:')
try:
    functionTwoParameters(a=1)
except TypeError as e:
    print(repr(e))
print('Call a function with two parameters with three keyword arguments:')
try:
    functionTwoParameters(a=1, b=2, c=3)
except TypeError as e:
    print(repr(e))

class MethodContainer:

    def methodNoParameters(self):
        if False:
            i = 10
            return i + 15
        pass

    def methodOneParameter(self, a):
        if False:
            print('Hello World!')
        print(a)

    def methodTwoParameters(self, a, b):
        if False:
            i = 10
            return i + 15
        print(a, b)
obj = MethodContainer()
print('Call a method with no parameters with a plain argument:')
try:
    obj.methodNoParameters(1)
except TypeError as e:
    print(repr(e))
print('Call a method with no parameters with a keyword argument:')
try:
    obj.methodNoParameters(z=1)
except TypeError as e:
    print(repr(e))
print('Call a method with one parameter with two plain arguments:')
try:
    obj.methodOneParameter(1, 1)
except TypeError as e:
    print(repr(e))
print('Call a method with two parameters with three plain arguments:')
try:
    obj.methodTwoParameters(1, 2, 3)
except TypeError as e:
    print(repr(e))
print('Call a method with two parameters with one plain argument:')
try:
    obj.methodTwoParameters(1)
except TypeError as e:
    print(repr(e))
print('Call a method with two parameters with one keyword argument:')
try:
    obj.methodTwoParameters(a=1)
except TypeError as e:
    print(repr(e))
print('Call a method with two parameters with three keyword arguments:')
try:
    obj.methodTwoParameters(a=1, b=2, c=3)
except TypeError as e:
    print(repr(e))

def functionPosBothStarArgs(a, b, c, *l, **d):
    if False:
        while True:
            i = 10
    print(a, b, c, l, d)
l = [2]
d = {'other': 7}
print('Call a function with both star arguments and too little arguments:')
try:
    functionPosBothStarArgs(1, *l, **d)
except TypeError as e:
    print(repr(e))
print('Call a function with defaults with too little arguments:')

def functionWithDefaults(a, b, c, d=3):
    if False:
        for i in range(10):
            print('nop')
    print(a, b, c, d)
try:
    functionWithDefaults(1)
except TypeError as e:
    print(repr(e))
print('Call a function with defaults with too many arguments:')
try:
    functionWithDefaults(1)
except TypeError as e:
    print(repr(e))
print('Complex calls with both invalid star list and star arguments:')
try:
    a = 1
    b = 2.0
    functionWithDefaults(1, *a, c=3, **b)
except TypeError as e:
    print(repr(e).replace('Value', '__main__.functionWithDefaults() argument'))
try:
    a = 1
    b = 2.0
    functionWithDefaults(1, *a, **b)
except TypeError as e:
    print(repr(e).replace('Value', '__main__.functionWithDefaults() argument'))
try:
    a = 1
    b = 2.0
    functionWithDefaults(*a, c=1, **b)
except TypeError as e:
    print(repr(e))
try:
    a = 1
    b = 2.0
    functionWithDefaults(*a, **b)
except TypeError as e:
    print(repr(e))
print('Complex call with both invalid star list argument:')
try:
    a = 1
    functionWithDefaults(*a)
except TypeError as e:
    print(repr(e))
try:
    a = 1
    MethodContainer(*a)
except TypeError as e:
    print(repr(e))
try:
    a = 1
    MethodContainer()(*a)
except TypeError as e:
    print(repr(e))
try:
    a = 1
    MethodContainer.methodTwoParameters(*a)
except TypeError as e:
    print(repr(e))
try:
    a = 1
    None(*a)
except TypeError as e:
    print(repr(e))
try:
    a = 1
    None(**a)
except TypeError as e:
    print(repr(e))
print('Call object with name as both keyword and in star dict argument:')
try:
    a = {'a': 3}
    None(a=2, **a)
except TypeError as e:
    print(repr(e))
print('Call function with only defaulted value given as keyword argument:')

def functionwithTwoArgsOneDefaulted(a, b=5):
    if False:
        while True:
            i = 10
    pass
try:
    functionwithTwoArgsOneDefaulted(b=12)
except TypeError as e:
    print(repr(e))