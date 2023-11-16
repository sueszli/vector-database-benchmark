""" Test disposing of app Components.
flexx/event/tests/test_disposing.py is focused on Component disposing by itself.
The tests in this module focus on app Components.
"""
import gc
import sys
import weakref
import asyncio
from pscript import this_is_js
from flexx import app, event
from flexx.util.testing import run_tests_if_main, raises, skipif, skip
from flexx.app.live_tester import run_live, roundtrip, launch
from flexx.event import loop
from flexx.app import PyComponent, JsComponent

def setup_module():
    if False:
        for i in range(10):
            print('nop')
    if sys.version_info > (3, 8) and sys.platform.startswith('win'):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    app.manager._clear_old_pending_sessions(1)

class MyPyComponent(PyComponent):

    def _dispose(self):
        if False:
            for i in range(10):
                print('nop')
        print('disposing', self.id)
        super()._dispose()

class MyJsComponent(JsComponent):

    def _dispose(self):
        if False:
            i = 10
            return i + 15
        print('disposing', self.id)
        super()._dispose()

def check_alive(s, id1, id2):
    if False:
        print('Hello World!')
    print(getattr(s.get_component_instance(id1), 'id', None))
    print(getattr(s.get_component_instance(id2), 'id', None))
    s.send_command('EVAL', 'flexx.s1.instances.%s && flexx.s1.instances.%s.id || null' % (id1, id1))
    s.send_command('EVAL', 'flexx.s1.instances.%s && flexx.s1.instances.%s.id || null' % (id2, id2))

@run_live
async def test_dispose_PyComponent1():
    """
    MyPyComponent_2
    MyPyComponent_3
    disposing MyPyComponent_2
    disposing MyPyComponent_3
    None
    None
    done
    ----------
    MyPyComponent_2
    MyPyComponent_3
    null
    null
    done
    """
    (c, s) = launch(PyComponent)
    with c:
        c1 = MyPyComponent()
        c2 = MyPyComponent()
    await roundtrip(s)
    (c1_id, c2_id) = (c1.id, c2.id)
    check_alive(s, c1_id, c2_id)
    await roundtrip(s)
    c1.dispose()
    c2.dispose()
    await roundtrip(s)
    await roundtrip(s)
    check_alive(s, c1_id, c2_id)
    await roundtrip(s)
    print('done')
    s.send_command('EVAL', '"done"')
    await roundtrip(s)

@run_live
async def test_dispose_PyComponent2():
    """
    MyPyComponent_2
    MyPyComponent_3
    disposing MyPyComponent_2
    disposing MyPyComponent_3
    None
    None
    done
    ----------
    MyPyComponent_2
    MyPyComponent_3
    null
    null
    done
    """
    (c, s) = launch(PyComponent)
    with c:
        c1 = MyPyComponent()
        c2 = MyPyComponent()
    (c1_id, c2_id) = (c1.id, c2.id)
    await roundtrip(s)
    check_alive(s, c1_id, c2_id)
    await roundtrip(s)
    del c1, c2
    gc.collect()
    await roundtrip(s)
    check_alive(s, c1_id, c2_id)
    await roundtrip(s)
    print('done')
    s.send_command('EVAL', '"done"')
    await roundtrip(s)

@skipif('__pypy__' in sys.builtin_module_names, reason='pypy gc works different')
@run_live
async def test_dispose_PyComponent3():
    """
    done
    disposing MyPyComponent_2
    disposing MyPyComponent_3
    ----------
    ? Cannot dispose a PyComponent from JS
    ? Cannot dispose a PyComponent from JS
    done
    """
    (c, s) = launch(PyComponent)
    with c:
        c1 = MyPyComponent()
        c2 = MyPyComponent()
    (c1_id, c2_id) = (c1.id, c2.id)
    await roundtrip(s)
    s.send_command('INVOKE', c1.id, 'dispose', [])
    s.send_command('INVOKE', c2.id, 'dispose', [])
    await roundtrip(s)
    print('done')
    s.send_command('EVAL', '"done"')
    del c1, c2
    gc.collect()
    await roundtrip(s)

@run_live
async def test_dispose_JsComponent1():
    """
    MyJsComponent_2
    MyJsComponent_3
    None
    None
    done
    ----------
    MyJsComponent_2
    MyJsComponent_3
    disposing MyJsComponent_2
    disposing MyJsComponent_3
    null
    null
    done
    """
    (c, s) = launch(PyComponent)
    with c:
        c1 = MyJsComponent()
        c2 = MyJsComponent()
    (c1_id, c2_id) = (c1.id, c2.id)
    await roundtrip(s)
    check_alive(s, c1_id, c2_id)
    await roundtrip(s)
    c1.dispose()
    c2.dispose()
    await roundtrip(s)
    check_alive(s, c1_id, c2_id)
    await roundtrip(s)
    print('done')
    s.send_command('EVAL', '"done"')
    await roundtrip(s)

@run_live
async def test_dispose_JsComponent2():
    """
    MyJsComponent_2
    MyJsComponent_3
    None
    None
    done
    ----------
    MyJsComponent_2
    MyJsComponent_3
    MyJsComponent_2
    MyJsComponent_3
    done
    """
    (c, s) = launch(PyComponent)
    with c:
        c1 = MyJsComponent()
        c2 = MyJsComponent()
    (c1_id, c2_id) = (c1.id, c2.id)
    await roundtrip(s)
    check_alive(s, c1_id, c2_id)
    await roundtrip(s)
    del c1, c2
    gc.collect()
    await roundtrip(s)
    check_alive(s, c1_id, c2_id)
    await roundtrip(s)
    print('done')
    s.send_command('EVAL', '"done"')
    await roundtrip(s)

@run_live
async def test_dispose_JsComponent3():
    """
    MyJsComponent_2
    MyJsComponent_3
    None
    None
    done
    ----------
    MyJsComponent_2
    MyJsComponent_3
    disposing MyJsComponent_2
    disposing MyJsComponent_3
    null
    null
    done
    """
    (c, s) = launch(PyComponent)
    with c:
        c1 = MyJsComponent()
        c2 = MyJsComponent()
    (c1_id, c2_id) = (c1.id, c2.id)
    await roundtrip(s)
    check_alive(s, c1_id, c2_id)
    await roundtrip(s)
    s.send_command('INVOKE', c1.id, 'dispose', [])
    s.send_command('INVOKE', c2.id, 'dispose', [])
    await roundtrip(s)
    check_alive(s, c1_id, c2_id)
    await roundtrip(s)
    print('done')
    s.send_command('EVAL', '"done"')
    await roundtrip(s)
run_tests_if_main()