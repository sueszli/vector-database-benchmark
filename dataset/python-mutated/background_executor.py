import time
import random
import concurrent.futures
from h2o_wave import main, app, Q, ui

def blocking_function(secs) -> str:
    if False:
        for i in range(10):
            print('nop')
    time.sleep(secs)
    return f'Done waiting for {secs} seconds!'

@app('/demo')
async def serve(q: Q):
    if q.args.start:
        q.page['form'] = ui.form_card(box='1 1 6 2', items=[ui.progress('Running...')])
        await q.page.save()
        seconds = random.randint(1, 6)
        with concurrent.futures.ThreadPoolExecutor() as pool:
            message = await q.exec(pool, blocking_function, seconds)
        q.page['form'] = ui.form_card(box='1 1 6 1', items=[ui.message_bar('info', message)])
        await q.page.save()
    else:
        q.page['form'] = ui.form_card(box='1 1 2 1', items=[ui.button(name='start', label='Start')])
        await q.page.save()