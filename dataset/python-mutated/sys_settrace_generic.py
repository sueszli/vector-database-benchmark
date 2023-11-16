print('Now comes the language constructions tests.')

def test_func():
    if False:
        return 10

    def test_sub_func():
        if False:
            return 10
        print('test_function')
    test_sub_func()

def test_closure(msg):
    if False:
        return 10

    def make_closure():
        if False:
            return 10
        print(msg)
    return make_closure

def test_exception():
    if False:
        return 10
    try:
        raise Exception('test_exception')
    except Exception:
        pass
    finally:
        pass

def test_listcomp():
    if False:
        while True:
            i = 10
    print('test_listcomp', [x for x in range(3)])

def test_lambda():
    if False:
        while True:
            i = 10
    func_obj_1 = lambda a, b: a + b
    print(func_obj_1(10, 20))

def test_import():
    if False:
        i = 10
        return i + 15
    from sys_settrace_subdir import sys_settrace_importme
    sys_settrace_importme.dummy()
    sys_settrace_importme.saysomething()

class TLClass:

    def method():
        if False:
            for i in range(10):
                print('nop')
        pass
    pass

def test_class():
    if False:
        for i in range(10):
            print('nop')

    class TestClass:
        __anynum = -9

        def method(self):
            if False:
                print('Hello World!')
            print('test_class_method')
            self.__anynum += 1

        def prprty_getter(self):
            if False:
                print('Hello World!')
            return self.__anynum

        def prprty_setter(self, what):
            if False:
                return 10
            self.__anynum = what
        prprty = property(prprty_getter, prprty_setter)
    cls = TestClass()
    cls.method()
    print('test_class_property', cls.prprty)
    cls.prprty = 12
    print('test_class_property', cls.prprty)

def run_tests():
    if False:
        while True:
            i = 10
    test_func()
    test_closure_inst = test_closure('test_closure')
    test_closure_inst()
    test_exception()
    test_listcomp()
    test_lambda()
    test_class()
    test_import()
print("And it's done!")