"""Example shows the recommended way of how to run Kivy with the Python built
in asyncio event loop as just another async coroutine.
"""
import asyncio
from kivy.app import App
from kivy.lang.builder import Builder
kv = '\nBoxLayout:\n    orientation: \'vertical\'\n    BoxLayout:\n        ToggleButton:\n            id: btn1\n            group: \'a\'\n            text: \'Sleeping\'\n            allow_no_selection: False\n            on_state: if self.state == \'down\': label.status = self.text\n        ToggleButton:\n            id: btn2\n            group: \'a\'\n            text: \'Swimming\'\n            allow_no_selection: False\n            on_state: if self.state == \'down\': label.status = self.text\n        ToggleButton:\n            id: btn3\n            group: \'a\'\n            text: \'Reading\'\n            allow_no_selection: False\n            state: \'down\'\n            on_state: if self.state == \'down\': label.status = self.text\n    Label:\n        id: label\n        status: \'Reading\'\n        text: \'Beach status is "{}"\'.format(self.status)\n'

class AsyncApp(App):
    other_task = None

    def build(self):
        if False:
            i = 10
            return i + 15
        return Builder.load_string(kv)

    def app_func(self):
        if False:
            return 10
        'This will run both methods asynchronously and then block until they\n        are finished\n        '
        self.other_task = asyncio.ensure_future(self.waste_time_freely())

        async def run_wrapper():
            await self.async_run(async_lib='asyncio')
            print('App done')
            self.other_task.cancel()
        return asyncio.gather(run_wrapper(), self.other_task)

    async def waste_time_freely(self):
        """This method is also run by the asyncio loop and periodically prints
        something.
        """
        try:
            i = 0
            while True:
                if self.root is not None:
                    status = self.root.ids.label.status
                    print('{} on the beach'.format(status))
                    if self.root.ids.btn1.state != 'down' and i >= 2:
                        i = 0
                        print('Yawn, getting tired. Going to sleep')
                        self.root.ids.btn1.trigger_action()
                i += 1
                await asyncio.sleep(2)
        except asyncio.CancelledError as e:
            print('Wasting time was canceled', e)
        finally:
            print('Done wasting time')
if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    loop.run_until_complete(AsyncApp().app_func())
    loop.close()