""" Demonstration of how to register event callbacks using an adaptation
of the color_scatter example from the bokeh gallery
"""
from __future__ import annotations
import numpy as np
from bokeh import events
from bokeh.io import curdoc, show
from bokeh.layouts import column, row
from bokeh.models import Button, CustomJS, Div, TextInput
from bokeh.plotting import figure

def display_event(div: Div, attributes: list[str]=[]) -> CustomJS:
    if False:
        print('Hello World!')
    '\n    Function to build a suitable CustomJS to display the current event\n    in the div model.\n    '
    style = 'float: left; clear: left; font-size: 13px'
    return CustomJS(args=dict(div=div), code=f"""\n        const attrs = {attributes};\n        const args = [];\n        for (let i = 0; i < attrs.length; i++) {{\n            const val = JSON.stringify(cb_obj[attrs[i]], function(key, val) {{\n                return val.toFixed ? Number(val.toFixed(2)) : val;\n            }})\n            args.push(attrs[i] + '=' + val)\n        }}\n        const line = "<span style={style!r}><b>" + cb_obj.event_name + "</b>(" + args.join(", ") + ")</span>\\n";\n        const text = div.text.concat(line);\n        const lines = text.split("\\n")\n        if (lines.length > 35)\n            lines.shift();\n        div.text = lines.join("\\n");\n    """)
N = 4000
x = np.random.random(size=N) * 100
y = np.random.random(size=N) * 100
radii = np.random.random(size=N) * 1.5
colors = np.array([(r, g, 150) for (r, g) in zip(50 + 2 * x, 30 + 2 * y)], dtype='uint8')
p = figure(tools='pan,wheel_zoom,zoom_in,zoom_out,reset,tap,lasso_select,box_select,box_zoom,undo,redo')
p.circle(x, y, radius=radii, fill_color=colors, fill_alpha=0.6, line_color=None)
div = Div(width=1000)
button = Button(label='Button', button_type='success', width=300)
text_input = TextInput(placeholder='Input a value and press Enter ...', width=300)
layout = column(button, text_input, row(p, div))
button.js_on_event(events.ButtonClick, display_event(div))
text_input.js_on_event(events.ValueSubmit, display_event(div, ['value']))
p.js_on_event(events.LODStart, display_event(div))
p.js_on_event(events.LODEnd, display_event(div))
point_attributes = ['x', 'y', 'sx', 'sy']
p.js_on_event(events.Tap, display_event(div, attributes=point_attributes))
p.js_on_event(events.DoubleTap, display_event(div, attributes=point_attributes))
p.js_on_event(events.Press, display_event(div, attributes=point_attributes))
p.js_on_event(events.PressUp, display_event(div, attributes=point_attributes))
p.js_on_event(events.MouseWheel, display_event(div, attributes=[*point_attributes, 'delta']))
p.js_on_event(events.MouseEnter, display_event(div, attributes=point_attributes))
p.js_on_event(events.MouseLeave, display_event(div, attributes=point_attributes))
pan_attributes = [*point_attributes, 'delta_x', 'delta_y']
p.js_on_event(events.Pan, display_event(div, attributes=pan_attributes))
p.js_on_event(events.PanStart, display_event(div, attributes=point_attributes))
p.js_on_event(events.PanEnd, display_event(div, attributes=point_attributes))
pinch_attributes = [*point_attributes, 'scale']
p.js_on_event(events.Pinch, display_event(div, attributes=pinch_attributes))
p.js_on_event(events.PinchStart, display_event(div, attributes=point_attributes))
p.js_on_event(events.PinchEnd, display_event(div, attributes=point_attributes))
p.js_on_event(events.RangesUpdate, display_event(div, attributes=['x0', 'x1', 'y0', 'y1']))
p.js_on_event(events.SelectionGeometry, display_event(div, attributes=['geometry', 'final']))
curdoc().on_event(events.DocumentReady, display_event(div))
show(layout)