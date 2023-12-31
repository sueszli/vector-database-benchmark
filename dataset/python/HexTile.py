from bokeh.io import curdoc, show
from bokeh.models import ColumnDataSource, Grid, HexTile, LinearAxis, Plot

source = ColumnDataSource(dict(
        q=[0,  0, -1, -1,  1, 1, 0],
        r=[0, -1,  0,  1, -1, 0, 1],
    ),
)

plot = Plot(
    title=None, width=300, height=300,
    min_border=0, toolbar_location=None)

glyph = HexTile(q="q", r="r", size=1, fill_color="#fb9a99", line_color="white")
plot.add_glyph(source, glyph)

xaxis = LinearAxis()
plot.add_layout(xaxis, 'below')

yaxis = LinearAxis()
plot.add_layout(yaxis, 'left')

plot.add_layout(Grid(dimension=0, ticker=xaxis.ticker))
plot.add_layout(Grid(dimension=1, ticker=yaxis.ticker))

curdoc().add_root(plot)

show(plot)
