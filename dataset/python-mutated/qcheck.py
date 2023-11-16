from __future__ import annotations
import datetime
import random
import re
import sys
import traceback
sys.path[0:0] = ['']
from bson.dbref import DBRef
from bson.objectid import ObjectId
from bson.son import SON
gen_target = 100
reduction_attempts = 10
examples = 5

def lift(value):
    if False:
        while True:
            i = 10
    return lambda : value

def choose_lifted(generator_list):
    if False:
        return 10
    return lambda : random.choice(generator_list)

def my_map(generator, function):
    if False:
        for i in range(10):
            print('nop')
    return lambda : function(generator())

def choose(list):
    if False:
        for i in range(10):
            print('nop')
    return lambda : random.choice(list)()

def gen_range(start, stop):
    if False:
        while True:
            i = 10
    return lambda : random.randint(start, stop)

def gen_int():
    if False:
        while True:
            i = 10
    max_int = 2147483647
    return lambda : random.randint(-max_int - 1, max_int)

def gen_float():
    if False:
        i = 10
        return i + 15
    return lambda : (random.random() - 0.5) * sys.maxsize

def gen_boolean():
    if False:
        while True:
            i = 10
    return lambda : random.choice([True, False])

def gen_printable_char():
    if False:
        i = 10
        return i + 15
    return lambda : chr(random.randint(32, 126))

def gen_printable_string(gen_length):
    if False:
        while True:
            i = 10
    return lambda : ''.join(gen_list(gen_printable_char(), gen_length)())

def gen_char(set=None):
    if False:
        for i in range(10):
            print('nop')
    return lambda : bytes([random.randint(0, 255)])

def gen_string(gen_length):
    if False:
        while True:
            i = 10
    return lambda : b''.join(gen_list(gen_char(), gen_length)())

def gen_unichar():
    if False:
        for i in range(10):
            print('nop')
    return lambda : chr(random.randint(1, 4095))

def gen_unicode(gen_length):
    if False:
        for i in range(10):
            print('nop')
    return lambda : ''.join([x for x in gen_list(gen_unichar(), gen_length)() if x not in '.$'])

def gen_list(generator, gen_length):
    if False:
        print('Hello World!')
    return lambda : [generator() for _ in range(gen_length())]

def gen_datetime():
    if False:
        return 10
    return lambda : datetime.datetime(random.randint(1970, 2037), random.randint(1, 12), random.randint(1, 28), random.randint(0, 23), random.randint(0, 59), random.randint(0, 59), random.randint(0, 999) * 1000)

def gen_dict(gen_key, gen_value, gen_length):
    if False:
        i = 10
        return i + 15

    def a_dict(gen_key, gen_value, length):
        if False:
            while True:
                i = 10
        result = {}
        for _ in range(length):
            result[gen_key()] = gen_value()
        return result
    return lambda : a_dict(gen_key, gen_value, gen_length())

def gen_regexp(gen_length):
    if False:
        i = 10
        return i + 15

    def pattern():
        if False:
            for i in range(10):
                print('nop')
        return ''.join(gen_list(choose_lifted('a'), gen_length)())

    def gen_flags():
        if False:
            return 10
        flags = 0
        if random.random() > 0.5:
            flags = flags | re.IGNORECASE
        if random.random() > 0.5:
            flags = flags | re.MULTILINE
        if random.random() > 0.5:
            flags = flags | re.VERBOSE
        return flags
    return lambda : re.compile(pattern(), gen_flags())

def gen_objectid():
    if False:
        return 10
    return lambda : ObjectId()

def gen_dbref():
    if False:
        return 10
    collection = gen_unicode(gen_range(0, 20))
    return lambda : DBRef(collection(), gen_mongo_value(1, True)())

def gen_mongo_value(depth, ref):
    if False:
        print('Hello World!')
    choices = [gen_unicode(gen_range(0, 50)), gen_printable_string(gen_range(0, 50)), my_map(gen_string(gen_range(0, 1000)), bytes), gen_int(), gen_float(), gen_boolean(), gen_datetime(), gen_objectid(), lift(None)]
    if ref:
        choices.append(gen_dbref())
    if depth > 0:
        choices.append(gen_mongo_list(depth, ref))
        choices.append(gen_mongo_dict(depth, ref))
    return choose(choices)

def gen_mongo_list(depth, ref):
    if False:
        print('Hello World!')
    return gen_list(gen_mongo_value(depth - 1, ref), gen_range(0, 10))

def gen_mongo_dict(depth, ref=True):
    if False:
        for i in range(10):
            print('nop')
    return my_map(gen_dict(gen_unicode(gen_range(0, 20)), gen_mongo_value(depth - 1, ref), gen_range(0, 10)), SON)

def simplify(case):
    if False:
        return 10
    if isinstance(case, SON) and '$ref' not in case:
        simplified = SON(case)
        if random.choice([True, False]):
            simplified_keys = list(simplified)
            if not len(simplified_keys):
                return (False, case)
            simplified.pop(random.choice(simplified_keys))
            return (True, simplified)
        else:
            simplified_items = list(simplified.items())
            if not len(simplified_items):
                return (False, case)
            (key, value) = random.choice(simplified_items)
            (success, value) = simplify(value)
            simplified[key] = value
            return (success, success and simplified or case)
    if isinstance(case, list):
        simplified = list(case)
        if random.choice([True, False]):
            if not len(simplified):
                return (False, case)
            simplified.pop(random.randrange(len(simplified)))
            return (True, simplified)
        else:
            if not len(simplified):
                return (False, case)
            index = random.randrange(len(simplified))
            (success, value) = simplify(simplified[index])
            simplified[index] = value
            return (success, success and simplified or case)
    return (False, case)

def reduce(case, predicate, reductions=0):
    if False:
        i = 10
        return i + 15
    for _ in range(reduction_attempts):
        (reduced, simplified) = simplify(case)
        if reduced and (not predicate(simplified)):
            return reduce(simplified, predicate, reductions + 1)
    return (reductions, case)

def isnt(predicate):
    if False:
        i = 10
        return i + 15
    return lambda x: not predicate(x)

def check(predicate, generator):
    if False:
        i = 10
        return i + 15
    counter_examples = []
    for _ in range(gen_target):
        case = generator()
        try:
            if not predicate(case):
                reduction = reduce(case, predicate)
                counter_examples.append('after {} reductions: {!r}'.format(*reduction))
        except:
            counter_examples.append(f'{case!r} : {traceback.format_exc()}')
    return counter_examples

def check_unittest(test, predicate, generator):
    if False:
        while True:
            i = 10
    counter_examples = check(predicate, generator)
    if counter_examples:
        failures = len(counter_examples)
        message = '\n'.join(['    -> %s' % f for f in counter_examples[:examples]])
        message = 'found %d counter examples, displaying first %d:\n%s' % (failures, min(failures, examples), message)
        test.fail(message)