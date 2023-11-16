from direct.showbase.CountedResource import CountedResource

def test_CountedResource():
    if False:
        print('Hello World!')

    class MouseResource(CountedResource):
        """
        A simple class to demonstrate the acquisition of a resource.
        """

        @classmethod
        def acquire(cls):
            if False:
                while True:
                    i = 10
            super(MouseResource, cls).acquire()
            print('-- Acquire Mouse')

        @classmethod
        def release(cls):
            if False:
                for i in range(10):
                    print('nop')
            print('-- Release Mouse')
            super(MouseResource, cls).release()

        def __init__(self):
            if False:
                print('Hello World!')
            super(MouseResource, self).__init__()

        def __del__(self):
            if False:
                return 10
            super(MouseResource, self).__del__()

    class CursorResource(CountedResource):
        """
        A class to demonstrate how to implement a dependent
        resource.  Notice how this class also inherits from
        CountedResource.  Instead of subclassing MouseCounter,
        we will just acquire it in our __init__() and release
        it in our __del__().
        """

        @classmethod
        def acquire(cls):
            if False:
                i = 10
                return i + 15
            super(CursorResource, cls).acquire()
            print('-- Acquire Cursor')

        @classmethod
        def release(cls):
            if False:
                return 10
            print('-- Release Cursor')
            super(CursorResource, cls).release()

        def __init__(self):
            if False:
                print('Hello World!')
            self.__mouseResource = MouseResource()
            super(CursorResource, self).__init__()

        def __del__(self):
            if False:
                for i in range(10):
                    print('nop')
            super(CursorResource, self).__del__()
            del self.__mouseResource

    class InvalidResource(MouseResource):

        @classmethod
        def acquire(cls):
            if False:
                while True:
                    i = 10
            super(InvalidResource, cls).acquire()
            print('-- Acquire Invalid')

        @classmethod
        def release(cls):
            if False:
                i = 10
                return i + 15
            print('-- Release Invalid')
            super(InvalidResource, cls).release()
    print('\nAllocate Mouse')
    m = MouseResource()
    print('Free up Mouse')
    del m
    print('\nAllocate Cursor')
    c = CursorResource()
    print('Free up Cursor')
    del c
    print('\nAllocate Mouse then Cursor')
    m = MouseResource()
    c = CursorResource()
    print('Free up Cursor')
    del c
    print('Free up Mouse')
    del m
    print('\nAllocate Mouse then Cursor')
    m = MouseResource()
    c = CursorResource()
    print('Free up Mouse')
    del m
    print('Free up Cursor')
    del c
    print('\nAllocate Cursor then Mouse')
    c = CursorResource()
    m = MouseResource()
    print('Free up Mouse')
    del m
    print('Free up Cursor')
    del c
    print('\nAllocate Cursor then Mouse')
    c = CursorResource()
    m = MouseResource()
    print('Free up Cursor')
    del c
    try:
        print('\nAllocate Invalid')
        i = InvalidResource()
        print('Free up Invalid')
    except AssertionError as e:
        print(e)
    print('')
    print('Free up Mouse')
    del m

    def demoFunc():
        if False:
            for i in range(10):
                print('nop')
        print('\nAllocate Cursor within function')
        c = CursorResource()
        print('Cursor will be freed on function exit')
    demoFunc()