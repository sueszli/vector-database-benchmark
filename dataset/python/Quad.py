import numpy as np

from bokeh.io import curdoc, show
from bokeh.models import ColumnDataSource, Grid, LinearAxis, Plot, Quad

N = 9
x = np.linspace(-2, 2, N)
y = x**2

source = ColumnDataSource(dict(
        left=x,
        top=y,
        right=x-x**3/10 + 0.3,
        bottom=y-x**2/10 + 0.5,
    ),
)

plot = Plot(
    title=None, width=300, height=300,
    min_border=0, toolbar_location=None)

glyph = Quad(left="left", right="right", top="top", bottom="bottom", fill_color="#b3de69")
plot.add_glyph(source, glyph)

xaxis = LinearAxis()
plot.add_layout(xaxis, 'below')

yaxis = LinearAxis()
plot.add_layout(yaxis, 'left')

plot.add_layout(Grid(dimension=0, ticker=xaxis.ticker))
plot.add_layout(Grid(dimension=1, ticker=yaxis.ticker))

curdoc().add_root(plot)

show(plot)
