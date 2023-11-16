"""Example shows the recommended way of how to run Kivy with a trio
event loop as just another async coroutine.
"""
import trio
from kivy.app import App
from kivy.lang.builder import Builder
kv = '\nBoxLayout:\n    orientation: \'vertical\'\n    BoxLayout:\n        ToggleButton:\n            id: btn1\n            group: \'a\'\n            text: \'Sleeping\'\n            allow_no_selection: False\n            on_state: if self.state == \'down\': label.status = self.text\n        ToggleButton:\n            id: btn2\n            group: \'a\'\n            text: \'Swimming\'\n            allow_no_selection: False\n            on_state: if self.state == \'down\': label.status = self.text\n        ToggleButton:\n            id: btn3\n            group: \'a\'\n            text: \'Reading\'\n            allow_no_selection: False\n            state: \'down\'\n            on_state: if self.state == \'down\': label.status = self.text\n    Label:\n        id: label\n        status: \'Reading\'\n        text: \'Beach status is "{}"\'.format(self.status)\n'

class AsyncApp(App):
    nursery = None

    def build(self):
        if False:
            i = 10
            return i + 15
        return Builder.load_string(kv)

    async def app_func(self):
        """trio needs to run a function, so this is it. """
        async with trio.open_nursery() as nursery:
            'In trio you create a nursery, in which you schedule async\n            functions to be run by the nursery simultaneously as tasks.\n\n            This will run all two methods starting in random order\n            asynchronously and then block until they are finished or canceled\n            at the `with` level. '
            self.nursery = nursery

            async def run_wrapper():
                await self.async_run(async_lib='trio')
                print('App done')
                nursery.cancel_scope.cancel()
            nursery.start_soon(run_wrapper)
            nursery.start_soon(self.waste_time_freely)

    async def waste_time_freely(self):
        """This method is also run by trio and periodically prints something.
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
                await trio.sleep(2)
        except trio.Cancelled as e:
            print('Wasting time was canceled', e)
        finally:
            print('Done wasting time')
if __name__ == '__main__':
    trio.run(AsyncApp().app_func)