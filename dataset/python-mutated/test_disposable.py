from reactivex.disposable import BooleanDisposable, CompositeDisposable, Disposable, RefCountDisposable, SerialDisposable, SingleAssignmentDisposable

def test_Disposable_create():
    if False:
        for i in range(10):
            print('nop')

    def action():
        if False:
            for i in range(10):
                print('nop')
        pass
    disp = Disposable(action)
    assert disp

def test_Disposable_dispose():
    if False:
        while True:
            i = 10
    disposed = [False]

    def action():
        if False:
            for i in range(10):
                print('nop')
        disposed[0] = True
    d = Disposable(action)
    assert not disposed[0]
    d.dispose()
    assert disposed[0]

def test_emptydisposable():
    if False:
        i = 10
        return i + 15
    d = Disposable()
    assert d
    d.dispose()

def test_booleandisposable():
    if False:
        return 10
    d = BooleanDisposable()
    assert not d.is_disposed
    d.dispose()
    assert d.is_disposed
    d.dispose()
    assert d.is_disposed

def test_future_disposable_setnone():
    if False:
        print('Hello World!')
    d = SingleAssignmentDisposable()
    d.disposable = None
    assert d.disposable == None

def test_futuredisposable_disposeafterset():
    if False:
        print('Hello World!')
    d = SingleAssignmentDisposable()
    disposed = [False]

    def action():
        if False:
            print('Hello World!')
        disposed[0] = True
    dd = Disposable(action)
    d.disposable = dd
    assert dd == d.disposable
    assert not disposed[0]
    d.dispose()
    assert disposed[0]
    d.dispose()
    assert disposed[0]

def test_futuredisposable_disposebeforeset():
    if False:
        i = 10
        return i + 15
    disposed = [False]

    def dispose():
        if False:
            return 10
        disposed[0] = True
    d = SingleAssignmentDisposable()
    dd = Disposable(dispose)
    assert not disposed[0]
    d.dispose()
    assert not disposed[0]
    d.disposable = dd
    assert d.disposable == None
    assert disposed[0]
    d.dispose()
    assert disposed[0]

def test_groupdisposable_contains():
    if False:
        return 10
    d1 = Disposable()
    d2 = Disposable()
    g = CompositeDisposable(d1, d2)
    assert g.length == 2
    assert g.contains(d1)
    assert g.contains(d2)

def test_groupdisposable_add():
    if False:
        for i in range(10):
            print('nop')
    d1 = Disposable()
    d2 = Disposable()
    g = CompositeDisposable(d1)
    assert g.length == 1
    assert g.contains(d1)
    g.add(d2)
    assert g.length == 2
    assert g.contains(d2)

def test_groupdisposable_addafterdispose():
    if False:
        for i in range(10):
            print('nop')
    disp1 = [False]
    disp2 = [False]

    def action1():
        if False:
            i = 10
            return i + 15
        disp1[0] = True
    d1 = Disposable(action1)

    def action2():
        if False:
            for i in range(10):
                print('nop')
        disp2[0] = True
    d2 = Disposable(action2)
    g = CompositeDisposable(d1)
    assert g.length == 1
    g.dispose()
    assert disp1[0]
    assert g.length == 0
    g.add(d2)
    assert disp2[0]
    assert g.length == 0

def test_groupdisposable_remove():
    if False:
        return 10
    disp1 = [False]
    disp2 = [False]

    def action1():
        if False:
            while True:
                i = 10
        disp1[0] = True
    d1 = Disposable(action1)

    def action2():
        if False:
            for i in range(10):
                print('nop')
        disp2[0] = True
    d2 = Disposable(action2)
    g = CompositeDisposable(d1, d2)
    assert g.length == 2
    assert g.contains(d1)
    assert g.contains(d2)
    assert g.remove(d1)
    assert g.length == 1
    assert not g.contains(d1)
    assert g.contains(d2)
    assert disp1[0]
    assert g.remove(d2)
    assert not g.contains(d1)
    assert not g.contains(d2)
    assert disp2[0]
    disp3 = [False]

    def action3():
        if False:
            i = 10
            return i + 15
        disp3[0] = True
    d3 = Disposable(action3)
    assert not g.remove(d3)
    assert not disp3[0]

def test_groupdisposable_clear():
    if False:
        return 10
    disp1 = [False]
    disp2 = [False]

    def action1():
        if False:
            i = 10
            return i + 15
        disp1[0] = True
    d1 = Disposable(action1)

    def action2():
        if False:
            i = 10
            return i + 15
        disp2[0] = True
    d2 = Disposable(action2)
    g = CompositeDisposable(d1, d2)
    assert g.length == 2
    g.clear()
    assert disp1[0]
    assert disp2[0]
    assert not g.length
    disp3 = [False]

    def action3():
        if False:
            print('Hello World!')
        disp3[0] = True
    d3 = Disposable(action3)
    g.add(d3)
    assert not disp3[0]
    assert g.length == 1

def test_mutabledisposable_ctor_prop():
    if False:
        while True:
            i = 10
    m = SerialDisposable()
    assert not m.disposable

def test_mutabledisposable_replacebeforedispose():
    if False:
        print('Hello World!')
    disp1 = [False]
    disp2 = [False]
    m = SerialDisposable()

    def action1():
        if False:
            while True:
                i = 10
        disp1[0] = True
    d1 = Disposable(action1)
    m.disposable = d1
    assert d1 == m.disposable
    assert not disp1[0]

    def action2():
        if False:
            for i in range(10):
                print('nop')
        disp2[0] = True
    d2 = Disposable(action2)
    m.disposable = d2
    assert d2 == m.disposable
    assert disp1[0]
    assert not disp2[0]

def test_mutabledisposable_replaceafterdispose():
    if False:
        while True:
            i = 10
    disp1 = [False]
    disp2 = [False]
    m = SerialDisposable()
    m.dispose()

    def action1():
        if False:
            return 10
        disp1[0] = True
    d1 = Disposable(action1)
    m.disposable = d1
    assert m.disposable == None
    assert disp1[0]

    def action2():
        if False:
            for i in range(10):
                print('nop')
        disp2[0] = True
    d2 = Disposable(action2)
    m.disposable = d2
    assert m.disposable == None
    assert disp2[0]

def test_mutabledisposable_dispose():
    if False:
        print('Hello World!')
    disp = [False]
    m = SerialDisposable()

    def action():
        if False:
            for i in range(10):
                print('nop')
        disp[0] = True
    d = Disposable(action)
    m.disposable = d
    assert d == m.disposable
    assert not disp[0]
    m.dispose()
    assert disp[0]
    assert m.disposable == None

def test_refcountdisposable_singlereference():
    if False:
        return 10
    d = BooleanDisposable()
    r = RefCountDisposable(d)
    assert not d.is_disposed
    r.dispose()
    assert d.is_disposed
    r.dispose()
    assert d.is_disposed

def test_refcountdisposable_refcounting():
    if False:
        print('Hello World!')
    d = BooleanDisposable()
    r = RefCountDisposable(d)
    assert not d.is_disposed
    d1 = r.disposable
    d2 = r.disposable
    assert not d.is_disposed
    d1.dispose()
    assert not d.is_disposed
    d2.dispose()
    assert not d.is_disposed
    r.dispose()
    assert d.is_disposed
    d3 = r.disposable
    d3.dispose()

def test_refcountdisposable_primarydisposesfirst():
    if False:
        return 10
    d = BooleanDisposable()
    r = RefCountDisposable(d)
    assert not d.is_disposed
    d1 = r.disposable
    d2 = r.disposable
    assert not d.is_disposed
    d1.dispose()
    assert not d.is_disposed
    r.dispose()
    assert not d.is_disposed
    d2.dispose()
    assert d.is_disposed