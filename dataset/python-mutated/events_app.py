""" Demonstration Bokeh app of how to register event callbacks in both
Javascript and Python using an adaptation of the color_scatter example
from the bokeh gallery. This example extends the js_events.py example
with corresponding Python event callbacks.
"""
import numpy as np
from bokeh import events
from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models import Button, CustomJS, Div
from bokeh.plotting import figure

def display_event(div, attributes=[]):
    if False:
        return 10
    '\n    Function to build a suitable CustomJS to display the current event\n    in the div model.\n    '
    style = 'float: left; clear: left; font-size: 13px'
    return CustomJS(args=dict(div=div), code=f"""\n        const {{to_string}} = Bokeh.require("core/util/pretty")\n        const attrs = {attributes};\n        const args = [];\n        for (let i = 0; i<attrs.length; i++ ) {{\n            const val = to_string(cb_obj[attrs[i]], {{precision: 2}})\n            args.push(attrs[i] + '=' + val)\n        }}\n        const line = "<span style={style!r}><b>" + cb_obj.event_name + "</b>(" + args.join(", ") + ")</span>\\n";\n        const text = div.text.concat(line);\n        const lines = text.split("\\n")\n        if (lines.length > 35)\n            lines.shift();\n        div.text = lines.join("\\n");\n    """)

def print_event(attributes=[]):
    if False:
        print('Hello World!')
    '\n    Function that returns a Python callback to pretty print the events.\n    '

    def python_callback(event):
        if False:
            i = 10
            return i + 15
        cls_name = event.__class__.__name__
        attrs = ', '.join([f'{attr}={event.__dict__[attr]}' for attr in attributes])
        print(f'{cls_name}({attrs})')
    return python_callback
N = 4000
x = np.random.random(size=N) * 100
y = np.random.random(size=N) * 100
radii = np.random.random(size=N) * 1.5
colors = np.array([(r, g, 150) for (r, g) in zip(50 + 2 * x, 30 + 2 * y)], dtype='uint8')
p = figure(tools='pan,wheel_zoom,zoom_in,zoom_out,reset,tap,lasso_select,box_select,box_zoom,undo,redo')
p.circle(x, y, radius=radii, fill_color=colors, fill_alpha=0.6, line_color=None)
div = Div(width=1000)
button = Button(label='Button', button_type='success', width=300)
layout = column(button, row(p, div))
point_attributes = ['x', 'y', 'sx', 'sy']
pan_attributes = [*point_attributes, 'delta_x', 'delta_y']
pinch_attributes = [*point_attributes, 'scale']
wheel_attributes = [*point_attributes, 'delta']
button.js_on_event(events.ButtonClick, display_event(div))
p.js_on_event(events.LODStart, display_event(div))
p.js_on_event(events.LODEnd, display_event(div))
p.js_on_event(events.Tap, display_event(div, attributes=point_attributes))
p.js_on_event(events.DoubleTap, display_event(div, attributes=point_attributes))
p.js_on_event(events.Press, display_event(div, attributes=point_attributes))
p.js_on_event(events.MouseWheel, display_event(div, attributes=wheel_attributes))
p.js_on_event(events.MouseEnter, display_event(div, attributes=point_attributes))
p.js_on_event(events.MouseLeave, display_event(div, attributes=point_attributes))
p.js_on_event(events.Pan, display_event(div, attributes=pan_attributes))
p.js_on_event(events.PanStart, display_event(div, attributes=point_attributes))
p.js_on_event(events.PanEnd, display_event(div, attributes=point_attributes))
p.js_on_event(events.Pinch, display_event(div, attributes=pinch_attributes))
p.js_on_event(events.PinchStart, display_event(div, attributes=point_attributes))
p.js_on_event(events.PinchEnd, display_event(div, attributes=point_attributes))
p.js_on_event(events.SelectionGeometry, display_event(div, attributes=['geometry', 'final']))
p.js_on_event(events.RangesUpdate, display_event(div, attributes=['x0', 'x1', 'y0', 'y1']))
p.js_on_event(events.Reset, display_event(div))
button.on_event(events.ButtonClick, print_event())
p.on_event(events.LODStart, print_event())
p.on_event(events.LODEnd, print_event())
p.on_event(events.Tap, print_event(attributes=point_attributes))
p.on_event(events.DoubleTap, print_event(attributes=point_attributes))
p.on_event(events.Press, print_event(attributes=point_attributes))
p.on_event(events.MouseWheel, print_event(attributes=wheel_attributes))
p.on_event(events.MouseEnter, print_event(attributes=point_attributes))
p.on_event(events.MouseLeave, print_event(attributes=point_attributes))
p.on_event(events.Pan, print_event(attributes=pan_attributes))
p.on_event(events.PanStart, print_event(attributes=point_attributes))
p.on_event(events.PanEnd, print_event(attributes=point_attributes))
p.on_event(events.Pinch, print_event(attributes=pinch_attributes))
p.on_event(events.PinchStart, print_event(attributes=point_attributes))
p.on_event(events.PinchEnd, print_event(attributes=point_attributes))
p.on_event(events.RangesUpdate, print_event(attributes=['x0', 'x1', 'y0', 'y1']))
p.on_event(events.SelectionGeometry, print_event(attributes=['geometry', 'final']))
p.on_event(events.Reset, print_event())
curdoc().add_root(layout)