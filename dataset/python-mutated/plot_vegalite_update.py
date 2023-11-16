from h2o_wave import site, data, ui
import random
import time
page = site['/demo']
spec = '\n{\n  "description": "A simple bar plot with embedded data.",\n  "mark": "bar",\n  "encoding": {\n    "x": {"field": "a", "type": "ordinal"},\n    "y": {"field": "b", "type": "quantitative"}\n  }\n}\n'

def rnd():
    if False:
        print('Hello World!')
    return random.randint(1, 100)

def poll():
    if False:
        print('Hello World!')
    return [['A', rnd()], ['B', rnd()], ['C', rnd()], ['D', rnd()], ['E', rnd()], ['F', rnd()], ['G', rnd()], ['H', rnd()], ['I', rnd()]]
vis = page.add('external', ui.vega_card(box='1 1 2 4', title='Plot with external data', specification=spec, data=data(fields=['a', 'b'], rows=poll())))
page.save()
while True:
    time.sleep(1)
    vis.data = poll()
    page.save()