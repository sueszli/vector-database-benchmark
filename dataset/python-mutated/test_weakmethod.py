import gc

def test_weak_method_on_obj():
    if False:
        i = 10
        return i + 15
    from kivy.weakmethod import WeakMethod

    class SomeClass:

        def do_something(self):
            if False:
                return 10
            pass
    obj = SomeClass()
    weak_method = WeakMethod(obj.do_something)
    assert not weak_method.is_dead()
    assert weak_method() == obj.do_something
    assert weak_method == WeakMethod(obj.do_something)
    assert weak_method != WeakMethod(SomeClass().do_something)
    del obj
    gc.collect()
    assert weak_method.is_dead()
    assert weak_method() is None
    assert weak_method != WeakMethod(SomeClass().do_something)

def test_weak_method_func():
    if False:
        return 10
    from kivy.weakmethod import WeakMethod

    def do_something():
        if False:
            i = 10
            return i + 15
        pass
    weak_method = WeakMethod(do_something)
    assert not weak_method.is_dead()
    assert weak_method() == do_something
    assert weak_method == WeakMethod(do_something)
    del do_something
    gc.collect()
    assert not weak_method.is_dead()
    assert weak_method() is not None