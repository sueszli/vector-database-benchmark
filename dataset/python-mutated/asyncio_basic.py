"""Example shows the recommended way of how to run Kivy with the Python built
in asyncio event loop as just another async coroutine.
"""
import asyncio
from kivy.app import async_runTouchApp
from kivy.lang.builder import Builder
kv = '\nBoxLayout:\n    orientation: \'vertical\'\n    Button:\n        id: btn\n        text: \'Press me\'\n    BoxLayout:\n        Label:\n            id: label\n            text: \'Button is "{}"\'.format(btn.state)\n'

async def run_app_happily(root, other_task):
    """This method, which runs Kivy, is run by the asyncio loop as one of the
    coroutines.
    """
    await async_runTouchApp(root, async_lib='asyncio')
    print('App done')
    other_task.cancel()

async def waste_time_freely():
    """This method is also run by the asyncio loop and periodically prints
    something.
    """
    try:
        while True:
            print('Sitting on the beach')
            await asyncio.sleep(2)
    except asyncio.CancelledError as e:
        print('Wasting time was canceled', e)
    finally:
        print('Done wasting time')
if __name__ == '__main__':

    def root_func():
        if False:
            return 10
        'This will run both methods asynchronously and then block until they\n        are finished\n        '
        root = Builder.load_string(kv)
        other_task = asyncio.ensure_future(waste_time_freely())
        return asyncio.gather(run_app_happily(root, other_task), other_task)
    loop = asyncio.get_event_loop()
    loop.run_until_complete(root_func())
    loop.close()