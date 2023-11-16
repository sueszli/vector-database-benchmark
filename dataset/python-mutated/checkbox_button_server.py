from bokeh.io import curdoc
from bokeh.layouts import row
from bokeh.models import CategoricalColorMapper, CheckboxButtonGroup, ColumnDataSource
from bokeh.palettes import RdBu3
from bokeh.plotting import figure
x = [3, 4, 6, 12, 10, 1]
y = [7, 1, 3, 4, 1, 6]
z = ['red', 'red', 'red', 'blue', 'blue', 'blue']
source = ColumnDataSource(data=dict(x=x, y=y, z=z))
color_mapper = CategoricalColorMapper(factors=['red', 'blue'], palette=[RdBu3[2], RdBu3[0]])
plot_figure = figure(title='Checkbox Button Group', height=450, width=600, tools='save,reset', toolbar_location='below')
plot_figure.scatter('x', 'y', source=source, size=10, color={'field': 'z', 'transform': color_mapper})
checkbox_button = CheckboxButtonGroup(labels=['Show x-axis label', 'Show y-axis label'])

def checkbox_button_click(attr, old, new):
    if False:
        for i in range(10):
            print('nop')
    active_checkbox = checkbox_button.active
    if len(active_checkbox) != 0 and 0 in active_checkbox:
        plot_figure.xaxis.axis_label = 'X-Axis'
    else:
        plot_figure.xaxis.axis_label = None
    if len(active_checkbox) != 0 and 1 in active_checkbox:
        plot_figure.yaxis.axis_label = 'Y-Axis'
    else:
        plot_figure.yaxis.axis_label = None
checkbox_button.on_change('active', checkbox_button_click)
layout = row(checkbox_button, plot_figure)
curdoc().add_root(layout)
curdoc().title = 'Checkbox Button Bokeh Server'