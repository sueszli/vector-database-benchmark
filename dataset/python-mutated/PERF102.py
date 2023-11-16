some_dict = {'a': 12, 'b': 32, 'c': 44}

def f():
    if False:
        print('Hello World!')
    for (_, value) in some_dict.items():
        print(value)

def f():
    if False:
        i = 10
        return i + 15
    for (key, _) in some_dict.items():
        print(key)

def f():
    if False:
        i = 10
        return i + 15
    for (weird_arg_name, _) in some_dict.items():
        print(weird_arg_name)

def f():
    if False:
        print('Hello World!')
    for (name, (_, _)) in some_dict.items():
        print(name)

def f():
    if False:
        print('Hello World!')
    for (name, (value1, _)) in some_dict.items():
        print(name, value1)

def f():
    if False:
        return 10
    for ((key1, _), (_, _)) in some_dict.items():
        print(key1)

def f():
    if False:
        for i in range(10):
            print('nop')
    for ((_, (_, _)), (value, _)) in some_dict.items():
        print(value)

def f():
    if False:
        print('Hello World!')
    for ((_, key2), (value1, _)) in some_dict.items():
        print(key2, value1)

def f():
    if False:
        return 10
    for ((_, key2), (value1, _)) in some_dict.items():
        print(key2, value1)

def f():
    if False:
        for i in range(10):
            print('nop')
    for ((_, key2), (_, _)) in some_dict.items():
        print(key2)

def f():
    if False:
        return 10
    for ((_, _, _, variants), (r_language, _, _, _)) in some_dict.items():
        print(variants, r_language)

def f():
    if False:
        while True:
            i = 10
    for ((_, _, (_, variants)), (_, (_, (r_language, _)))) in some_dict.items():
        print(variants, r_language)

def f():
    if False:
        i = 10
        return i + 15
    for (key, value) in some_dict.items():
        print(key, value)

def f():
    if False:
        for i in range(10):
            print('nop')
    for (_, value) in some_dict.items(12):
        print(value)

def f():
    if False:
        i = 10
        return i + 15
    for key in some_dict.keys():
        print(key)

def f():
    if False:
        while True:
            i = 10
    for value in some_dict.values():
        print(value)

def f():
    if False:
        i = 10
        return i + 15
    for (name, (_, _)) in some_function().items():
        print(name)

def f():
    if False:
        i = 10
        return i + 15
    for (name, (_, _)) in some_function().some_attribute.items():
        print(name)

def f():
    if False:
        while True:
            i = 10
    for (name, unused_value) in some_dict.items():
        print(name)

def f():
    if False:
        return 10
    for (unused_name, value) in some_dict.items():
        print(value)

def _create_context(name_to_value):
    if False:
        print('Hello World!')
    for (B, D) in A.items():
        if (C := name_to_value.get(B.name)):
            A.run(B.set, C)