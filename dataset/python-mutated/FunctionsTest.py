""" Various kinds of functions with small specialties.

"""
from __future__ import print_function
import inspect
import sys
var_on_module_level = 1

def closureTest1(some_arg):
    if False:
        while True:
            i = 10
    x = 3

    def enclosed(_f='default_value'):
        if False:
            while True:
                i = 10
        return x
    return enclosed
print('Call closured function returning function:', closureTest1(some_arg='ignored')())

def closureTest2(some_arg):
    if False:
        for i in range(10):
            print('nop')

    def enclosed(_f='default_value'):
        if False:
            i = 10
            return i + 15
        return x
    x = 4
    return enclosed
print('Call closured function returning function:', closureTest2(some_arg='ignored')())

def defaultValueTest1(_no_default, some_default_constant=1):
    if False:
        return 10
    return some_default_constant
print('Call function with 2 parameters, one defaulted, and check that the default value is used:', defaultValueTest1('ignored'))

def defaultValueTest1a(_no_default, some_default_constant_1=1, some_default_constant_2=2):
    if False:
        for i in range(10):
            print('nop')
    return some_default_constant_2 - some_default_constant_1
print('Call function with 3 parameters, 2 defaulted, check they are used correctly:', defaultValueTest1a('ignored'))

def defaultValueTest2(_no_default, some_default_variable=var_on_module_level * 2):
    if False:
        while True:
            i = 10
    return some_default_variable
print('Call function with 2 parameters, 1 defaulted with an expression, check its result', defaultValueTest2('ignored'))
var_on_module_level = 2
print('Call function with 2 parameters, 1 defaulted with an expression, values have changed since, check its result', defaultValueTest2('ignored'))

def contractionTest():
    if False:
        i = 10
        return i + 15
    j = 2
    return [j + i for i in range(8)]
print('Call function that returns a list contraction:', contractionTest())

def defaultValueTest3a(_no_default, funced_defaulted=defaultValueTest2(var_on_module_level)):
    if False:
        while True:
            i = 10
    return [i + funced_defaulted for i in range(8)]
print('Call function that has a default value coming from a function call:', defaultValueTest3a('ignored'))

def defaultValueTest3b(_no_default, funced_defaulted=defaultValueTest2(var_on_module_level)):
    if False:
        i = 10
        return i + 15
    local_var = [funced_defaulted + i for i in range(8)]
    return local_var
print('Call function that returns a list contraction result via a local variable:', defaultValueTest3b('ignored'))

def defaultValueTest3c(_no_default, funced_defaulted=defaultValueTest2(var_on_module_level)):
    if False:
        i = 10
        return i + 15
    local_var = [[j + funced_defaulted + 1 for j in range(i)] for i in range(8)]
    return local_var
print('Call function that returns a nested list contraction with input from default parameter', defaultValueTest3c('ignored'))

def defaultValueTest4(_no_default, funced_defaulted=lambda x: x ** 2):
    if False:
        while True:
            i = 10
    return funced_defaulted(4)
print('Call function that returns value calculated by a lambda function as default parameter', defaultValueTest4('ignored'))

def defaultValueTest4a(_no_default, funced_defaulted=lambda x: x ** 2):
    if False:
        while True:
            i = 10
    c = 1
    d = funced_defaulted(1)
    r = (i + j + c + d for (i, j) in zip(range(8), range(9)))
    l = []
    for x in r:
        l.append(x)
    return l
print('Call function that has a lambda calculated default parameter and a generator expression', defaultValueTest4a('ignored'))

def defaultValueTest4b(_no_default, funced_defaulted=lambda x: x ** 3):
    if False:
        for i in range(10):
            print('nop')
    d = funced_defaulted(1)
    l = []
    for x in ((d + j for j in range(4)) for i in range(8)):
        for y in x:
            l.append(y)
    return l
print('Call function that has a lambda calculated default parameter and a nested generator expression', defaultValueTest4b('ignored'))

def defaultValueTest5(_no_default, tuple_defaulted=(1, 2, 3)):
    if False:
        for i in range(10):
            print('nop')
    return tuple_defaulted
print('Call function with default value that is a tuple', defaultValueTest5('ignored'))

def defaultValueTest6(_no_default, list_defaulted=[1, 2, 3]):
    if False:
        return 10
    return list_defaulted
print('Call function with default value that is a list', defaultValueTest6('ignored'))
x = len('hey')

def in_test(a):
    if False:
        i = 10
        return i + 15
    8 in a
    9 not in a
in_test([8])
try:
    in_test(9)
except TypeError:
    pass

def my_deco(function):
    if False:
        while True:
            i = 10

    def new_function(c, d):
        if False:
            while True:
                i = 10
        return function(d, c)
    return new_function

@my_deco
def decoriert(a, b):
    if False:
        while True:
            i = 10

    def subby(a):
        if False:
            while True:
                i = 10
        return 2 + a
    return 1 + subby(b)
print('Function with decoration', decoriert(3, 9))

def functionWithGlobalReturnValue():
    if False:
        print('Hello World!')
    global a
    return a
a = 'oh common'
some_constant_tuple = (2, 5, 7)
some_semiconstant_tuple = (2, 5, a)
f = a * 2
print(defaultValueTest1('ignored'))
module_level = 7
print(defaultValueTest2('also ignored'))

def starArgTest(a, b, c):
    if False:
        print('Hello World!')
    return (a, b, c)
print('Function called with star arg from tuple:')
star_list_arg = (11, 44, 77)
print(starArgTest(*star_list_arg))
print('Function called with star arg from list:')
star_list_arg = [7, 8, 9]
print(starArgTest(*star_list_arg))
star_dict_arg = {'a': 9, 'b': 3, 'c': 8}
print('Function called with star arg from dict')
print(starArgTest(**star_dict_arg))
lambda_func = lambda a, b: a < b
lambda_args = (8, 9)
print('Lambda function called with star args from tuple:')
print(lambda_func(*lambda_args))
print('Lambda function called with star args from list:')
lambda_args = [8, 7]
print(lambda_func(*lambda_args))
print('Generator function without context:')

def generator_without_context_function():
    if False:
        for i in range(10):
            print('nop')
    gen = (x for x in range(9))
    return tuple(gen)
print(generator_without_context_function())
print('Generator function with 2 iterateds:')

def generator_with_2_fors():
    if False:
        print('Hello World!')
    return tuple(((x, y) for x in range(2) for y in range(3)))
print(generator_with_2_fors())

def someYielder():
    if False:
        for i in range(10):
            print('nop')
    yield 1
    yield 2

def someYieldFunctionUser():
    if False:
        print('Hello World!')
    print('someYielder', someYielder())
    result = []
    for a in someYielder():
        result.append(a)
    return result
print('Function that uses some yielding function coroutine:')
print(someYieldFunctionUser())

def someLoopYielder():
    if False:
        i = 10
        return i + 15
    for i in (0, 1, 2):
        yield i

def someLoopYieldFunctionUser():
    if False:
        return 10
    result = []
    for a in someLoopYielder():
        result.append(a)
    return result
print('Function that uses some yielding function coroutine that loops:')
print(someLoopYieldFunctionUser())

def someGeneratorClosureUser():
    if False:
        return 10

    def someGenerator():
        if False:
            i = 10
            return i + 15

        def userOfGeneratorLocalVar():
            if False:
                for i in range(10):
                    print('nop')
            return x + 1
        x = 2
        yield userOfGeneratorLocalVar()
        yield 6
    gen = someGenerator()
    return [next(gen), next(gen)]
print('Function generator that uses a local function accessing its local variables to yield:')
print(someGeneratorClosureUser())

def someClosureUsingGeneratorUser():
    if False:
        print('Hello World!')
    offered = 7

    def someGenerator():
        if False:
            for i in range(10):
                print('nop')
        yield offered
    return next(someGenerator())
print('Function generator that yield from its closure:')
print(someClosureUsingGeneratorUser())
print('Function call with both star args and named args:')

def someFunction(a, b, c, d):
    if False:
        return 10
    print(a, b, c, d)
someFunction(a=1, b=2, **{'c': 3, 'd': 4})
print('Order of evaluation of function and args:')

def getFunction():
    if False:
        while True:
            i = 10
    print('getFunction', end='')

    def x(y, u, a, k):
        if False:
            i = 10
            return i + 15
        return (y, u, k, a)
    return x

def getPlainArg1():
    if False:
        print('Hello World!')
    print('getPlainArg1', end='')
    return 9

def getPlainArg2():
    if False:
        while True:
            i = 10
    print('getPlainArg2', end='')
    return 13

def getKeywordArg1():
    if False:
        print('Hello World!')
    print('getKeywordArg1', end='')
    return 'a'

def getKeywordArg2():
    if False:
        for i in range(10):
            print('nop')
    print('getKeywordArg2', end='')
    return 'b'
getFunction()(getPlainArg1(), getPlainArg2(), k=getKeywordArg1(), a=getKeywordArg2())
print()

def getListStarArg():
    if False:
        while True:
            i = 10
    print('getListStarArg', end='')
    return [1]

def getDictStarArg():
    if False:
        return 10
    print('getDictStarArg', end='')
    return {'k': 9}
print('Same with star args:')
getFunction()(getPlainArg1(), *getListStarArg(), a=getKeywordArg1(), **getDictStarArg())
print()
print('Dictionary creation order:')
d = {getKeywordArg1(): getPlainArg1(), getKeywordArg2(): getPlainArg2()}
print()
print('Throwing an exception to a generator function:')

def someGeneratorFunction():
    if False:
        while True:
            i = 10
    try:
        yield 1
        yield 2
    except Exception:
        yield 3
    yield 4
gen1 = someGeneratorFunction()
print('Fresh Generator Function throwing gives', end='')
try:
    (print(gen1.throw(ValueError)),)
except ValueError:
    print('exception indeed')
gen2 = someGeneratorFunction()
print('Used Generator Function throwing gives', end='')
next(gen2)
print(gen2.throw(ValueError), 'indeed')
gen3 = someGeneratorFunction()
print('Fresh Generator Function close gives', end='')
print(gen3.close())
gen4 = someGeneratorFunction()
print('Used Generator Function that mis-catches close gives', end='')
next(gen4)
try:
    print(gen4.close(), end='')
except RuntimeError:
    print('runtime exception indeed')
gen5 = someGeneratorFunction()
print('Used Generator Function close gives', end='')
next(gen5)
next(gen5)
next(gen5)
print(gen5.close(), end='')

def receivingGenerator():
    if False:
        return 10
    while True:
        a = (yield 4)
        yield a
print('Generator function that receives', end='')
gen6 = receivingGenerator()
print(next(gen6), end='')
print(gen6.send(5), end='')
print(gen6.send(6), end='')
print(gen6.send(7), end='')
print(gen6.send(8))
print('Generator function whose generator is copied', end='')

def generatorFunction():
    if False:
        return 10
    yield 1
    yield 2
gen7 = generatorFunction()
next(gen7)
gen8 = iter(gen7)
print(next(gen8))

def doubleStarArgs(*a, **d):
    if False:
        i = 10
        return i + 15
    return (a, d)
try:
    from UserDict import UserDict
except ImportError:
    print('Using Python3, making own non-dict dict:')

    class UserDict(dict):
        pass
print('Function that has keyword argument matching the list star arg name', end='')
print(doubleStarArgs(1, **UserDict(a=2)))

def generatorFunctionUnusedArg(_a):
    if False:
        i = 10
        return i + 15
    yield 1
generatorFunctionUnusedArg(3)

def closureHavingGenerator(arg):
    if False:
        print('Hello World!')

    def gen(_x=1):
        if False:
            i = 10
            return i + 15
        yield arg
    return gen()
print('Function generator that has a closure and default argument', end='')
print(list(closureHavingGenerator(3)))

def functionWithDualStarArgsAndKeywordsOnly(a1, a2, a3, a4, b):
    if False:
        return 10
    return (a1, a2, a3, a4, b)
l = [1, 2, 3]
d = {'b': 8}
print('Dual star args, but not positional call', functionWithDualStarArgsAndKeywordsOnly(*l, a4=1, **d))

def posDoubleStarArgsFunction(a, b, c, *l, **d):
    if False:
        for i in range(10):
            print('nop')
    return (a, b, c, l, d)
l = [2]
d = {'other': 7, 'c': 3}
print('Dual star args consuming function', posDoubleStarArgsFunction(1, *l, **d))
for value in sorted(dir()):
    main_value = getattr(sys.modules['__main__'], value)
    if inspect.isfunction(main_value):
        print(main_value, main_value.__code__.co_varnames[:main_value.__code__.co_argcount], inspect.getargs(main_value.__code__))