import asyncio
from reactpy import component, html, run

@component
def ButtonWithDelay(message, delay):
    if False:
        i = 10
        return i + 15

    async def handle_event(event):
        await asyncio.sleep(delay)
        print(message)
    return html.button({'on_click': handle_event}, message)

@component
def App():
    if False:
        for i in range(10):
            print('nop')
    return html.div(ButtonWithDelay('print 3 seconds later', delay=3), ButtonWithDelay('print immediately', delay=0))
run(App)