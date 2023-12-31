## Bokeh server for Tab
import numpy as np

from bokeh.io import curdoc
from bokeh.layouts import row
from bokeh.models import TabPanel, Tabs
from bokeh.plotting import figure

n = 1000
x = 3 + 3*np.random.standard_normal(n)
y = 3 + 3*np.random.standard_normal(n)

p1 = figure(background_fill_color='white', match_aspect=True)
p1.scatter(x, y, size=3, color="black")
tab1 = TabPanel(child=p1, title="Scatter Plot")
p1.grid.visible = False

p2 = figure(background_fill_color='white', match_aspect=True)
p2.hexbin(x,y, size=0.3, hover_alpha=0.5,line_color='grey')
tab2 = TabPanel(child=p2, title="Hexbin Plot")
p2.grid.visible = False

tabs = Tabs(tabs=[ tab1, tab2 ])

layout=row(tabs)

curdoc().add_root(layout)
curdoc().title = "Tab Bokeh Server"
