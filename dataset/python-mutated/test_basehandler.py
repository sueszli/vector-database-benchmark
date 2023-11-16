from telegram.ext._basehandler import BaseHandler
from tests.auxil.slots import mro_slots

class TestHandler:

    def test_slot_behaviour(self):
        if False:
            print('Hello World!')

        class SubclassHandler(BaseHandler):
            __slots__ = ()

            def __init__(self):
                if False:
                    print('Hello World!')
                super().__init__(lambda x: None)

            def check_update(self, update: object):
                if False:
                    i = 10
                    return i + 15
                pass
        inst = SubclassHandler()
        for attr in inst.__slots__:
            assert getattr(inst, attr, 'err') != 'err', f"got extra slot '{attr}'"
        assert len(mro_slots(inst)) == len(set(mro_slots(inst))), 'duplicate slot'

    def test_repr(self):
        if False:
            i = 10
            return i + 15

        async def some_func():
            return None

        class SubclassHandler(BaseHandler):
            __slots__ = ()

            def __init__(self):
                if False:
                    print('Hello World!')
                super().__init__(callback=some_func)

            def check_update(self, update: object):
                if False:
                    while True:
                        i = 10
                pass
        sh = SubclassHandler()
        assert repr(sh) == 'SubclassHandler[callback=TestHandler.test_repr.<locals>.some_func]'

    def test_repr_no_qualname(self):
        if False:
            for i in range(10):
                print('nop')

        class ClassBasedCallback:

            async def __call__(self, *args, **kwargs):
                pass

            def __repr__(self):
                if False:
                    i = 10
                    return i + 15
                return 'Repr of ClassBasedCallback'

        class SubclassHandler(BaseHandler):
            __slots__ = ()

            def __init__(self):
                if False:
                    while True:
                        i = 10
                super().__init__(callback=ClassBasedCallback())

            def check_update(self, update: object):
                if False:
                    return 10
                pass
        sh = SubclassHandler()
        assert repr(sh) == 'SubclassHandler[callback=Repr of ClassBasedCallback]'