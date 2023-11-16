from robot.api import logger
from robot.api.deco import keyword
from robot.libraries.BuiltIn import BuiltIn
ROBOT_AUTO_KEYWORDS = False
should_be_equal = BuiltIn().should_be_equal
log = logger.write

@keyword(name='User ${user} Selects ${item} From Webshop')
def user_selects_from_webshop(user, item):
    if False:
        for i in range(10):
            print('nop')
    log('This is always executed')
    return (user, item)

@keyword(name='${prefix:Given|When|Then} this "${item}" ${no good name for this arg ...}')
def this(ignored_prefix, item, somearg):
    if False:
        print('Hello World!')
    log('%s-%s' % (item, somearg))

@keyword(name='My embedded ${var}')
def my_embedded(var):
    if False:
        i = 10
        return i + 15
    should_be_equal(var, 'warrior')

@keyword(name='${x:x} gets ${y:\\w} from the ${z:.}')
def gets_from_the(x, y, z):
    if False:
        return 10
    should_be_equal('%s-%s-%s' % (x, y, z), 'x-y-z')

@keyword(name='${a}-lib-${b}')
def mult_match1(a, b):
    if False:
        print('Hello World!')
    log('%s-lib-%s' % (a, b))

@keyword(name='${a}+lib+${b}')
def mult_match2(a, b):
    if False:
        for i in range(10):
            print('nop')
    log('%s+lib+%s' % (a, b))

@keyword(name='${a}*lib*${b}')
def mult_match3(a, b):
    if False:
        return 10
    log('%s*lib*%s' % (a, b))

@keyword(name='I execute "${x:[^"]*}"')
def i_execute(x):
    if False:
        i = 10
        return i + 15
    should_be_equal(x, 'foo')

@keyword(name='I execute "${x:bar}" with "${y:...}"')
def i_execute_with(x, y):
    if False:
        return 10
    should_be_equal(x, 'bar')
    should_be_equal(y, 'zap')

@keyword(name='Result of ${a:\\d+} ${operator:[+-]} ${b:\\d+} is ${result}')
def result_of_is(a, operator, b, result):
    if False:
        i = 10
        return i + 15
    should_be_equal(eval('%s%s%s' % (a, operator, b)), float(result))

@keyword(name='I want ${integer:whatever} and ${string:everwhat} as variables')
def i_want_as_variables(integer, string):
    if False:
        i = 10
        return i + 15
    should_be_equal(integer, 42)
    should_be_equal(string, '42')

@keyword(name='Today is ${date:\\d{4}-\\d{2}-\\d{2}}')
def today_is(date):
    if False:
        for i in range(10):
            print('nop')
    should_be_equal(date, '2011-06-21')

@keyword(name='Today is ${day1:\\w{6,9}} and tomorrow is ${day2:\\w{6,9}}')
def today_is_and_tomorrow_is(day1, day2):
    if False:
        return 10
    should_be_equal(day1, 'Tuesday')
    should_be_equal(day2, 'Wednesday')

@keyword(name='Literal ${Curly:\\{} Brace')
def literal_opening_curly_brace(curly):
    if False:
        print('Hello World!')
    should_be_equal(curly, '{')

@keyword(name='Literal ${Curly:\\}} Brace')
def literal_closing_curly_brace(curly):
    if False:
        for i in range(10):
            print('nop')
    should_be_equal(curly, '}')

@keyword(name='Literal ${Curly:{}} Braces')
def literal_curly_braces(curly):
    if False:
        for i in range(10):
            print('nop')
    should_be_equal(curly, '{}')

@keyword(name='Custom Regexp With Escape Chars e.g. ${1E:\\\\}, ${2E:\\\\\\\\} and ${PATH:c:\\\\temp\\\\.*}')
def custom_regexp_with_escape_chars(e1, e2, path):
    if False:
        i = 10
        return i + 15
    should_be_equal(e1, '\\')
    should_be_equal(e2, '\\\\')
    should_be_equal(path, 'c:\\temp\\test.txt')

@keyword(name='Custom Regexp With ${escapes:\\\\\\}}')
def custom_regexp_with_escapes_1(escapes):
    if False:
        print('Hello World!')
    should_be_equal(escapes, '\\}')

@keyword(name='Custom Regexp With ${escapes:\\\\\\{}')
def custom_regexp_with_escapes_2(escapes):
    if False:
        while True:
            i = 10
    should_be_equal(escapes, '\\{')

@keyword(name='Custom Regexp With ${escapes:\\\\{}}')
def custom_regexp_with_escapes_3(escapes):
    if False:
        print('Hello World!')
    should_be_equal(escapes, '\\{}')

@keyword(name='Grouping ${x:Cu(st|ts)(om)?} ${y:Regexp\\(?erts\\)?}')
def grouping(x, y):
    if False:
        print('Hello World!')
    return f'{x}-{y}'

@keyword(name='Wrong ${number} of embedded ${args}')
def too_few_args_here(arg):
    if False:
        for i in range(10):
            print('nop')
    pass

@keyword(name='Optional non-${embedded} Args Are ${okay}')
def optional_args_are_okay(embedded=1, okay=2, extra=3):
    if False:
        i = 10
        return i + 15
    return (embedded, okay, extra)

@keyword(name='Varargs With ${embedded} Args Are ${okay}')
def varargs_are_okay(*args):
    if False:
        i = 10
        return i + 15
    return args

@keyword('It is ${vehicle:a (car|ship)}')
def same_name_1(vehicle):
    if False:
        return 10
    log(vehicle)

@keyword('It is ${animal:a (dog|cat)}')
def same_name_2(animal):
    if False:
        for i in range(10):
            print('nop')
    log(animal)

@keyword('It is ${animal:a (cat|cow)}')
def same_name_3(animal):
    if False:
        for i in range(10):
            print('nop')
    log(animal)

@keyword('It is totally ${same}')
def totally_same_1(arg):
    if False:
        while True:
            i = 10
    raise Exception('Not executed')

@keyword('It is totally ${same}')
def totally_same_2(arg):
    if False:
        while True:
            i = 10
    raise Exception('Not executed')

@keyword('Number of ${animals} should be')
def number_of_animals_should_be(animals, count, activity='walking'):
    if False:
        return 10
    log(f'{count} {animals} are {activity}')

@keyword('Conversion with embedded ${number} and normal')
def conversion_with_embedded_and_normal(num1: int, /, num2: int):
    if False:
        i = 10
        return i + 15
    assert num1 == num2 == 42