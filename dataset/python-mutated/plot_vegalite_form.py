from h2o_wave import site, data, ui
import random
import time
page = site['/demo']
spec = '\n{\n  "description": "A simple bar plot with embedded data.",\n  "mark": "bar",\n  "encoding": {\n    "x": {"field": "a", "type": "ordinal"},\n    "y": {"field": "b", "type": "quantitative"}\n  }\n}\n'

def poll():
    if False:
        i = 10
        return i + 15
    return [['A', rnd()], ['B', rnd()], ['C', rnd()], ['D', rnd()], ['E', rnd()], ['F', rnd()], ['G', rnd()], ['H', rnd()], ['I', rnd()]]

def rnd():
    if False:
        print('Hello World!')
    return random.randint(1, 100)
page['example'] = ui.form_card(box='1 1 -1 -1', items=[ui.text_xl('Example 1'), ui.vega_visualization(specification=spec, data=data(fields=['a', 'b'], rows=poll(), pack=True)), ui.text_xl('Example 2'), ui.vega_visualization(specification=spec, data=data(fields=['a', 'b'], rows=poll(), pack=True)), ui.text_xl('Example 3'), ui.vega_visualization(specification=spec, data=data(fields=['a', 'b'], rows=poll(), pack=True))])
page.save()