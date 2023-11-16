import asyncio
import time
import concurrent.futures
from threading import Event
from h2o_wave import main, app, Q, ui

def blocking_function(q: Q, loop: asyncio.AbstractEventLoop):
    if False:
        return 10
    count = 0
    total = 10
    future = None
    while count < total:
        if q.client.event.is_set():
            asyncio.ensure_future(show_cancel(q), loop=loop)
            return
        time.sleep(1)
        count += 1
        if not future or future.done():
            future = asyncio.ensure_future(update_ui(q, count / total), loop=loop)

async def show_cancel(q: Q):
    q.page['form'].progress.caption = 'Cancelled'
    await q.page.save()

async def update_ui(q: Q, value: int):
    q.page['form'].progress.value = value
    await q.page.save()

@app('/demo')
async def serve(q: Q):
    if not q.client.initialized:
        q.page['form'] = ui.form_card(box='1 1 3 2', items=[ui.buttons([ui.button(name='start_job', label='Start job'), ui.button(name='cancel', label='Cancel')]), ui.progress(name='progress', label='Progress', value=0)])
        q.client.initialized = True
    if q.args.start_job:
        loop = asyncio.get_event_loop()
        q.client.event = Event()
        with concurrent.futures.ThreadPoolExecutor() as pool:
            await q.exec(pool, blocking_function, q, loop)
    if q.args.cancel:
        q.client.event.set()
    await q.page.save()