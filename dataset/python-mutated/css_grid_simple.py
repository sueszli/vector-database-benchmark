from bokeh.io import save
from bokeh.models import LinearAxis, Plot, Range1d, Scatter
aliases = dict(a='above', b='below', l='left', r='right')

def make_figure(axes):
    if False:
        while True:
            i = 10
    xdr = Range1d(start=-1, end=1)
    ydr = Range1d(start=-1, end=1)
    plot = Plot(title=None, x_range=xdr, y_range=ydr, width=200, height=200, toolbar_location=None)
    plot.add_glyph(Scatter(x=0, y=0, size=100))
    for place in axes:
        plot.add_layout(LinearAxis(), aliases[place])
    return plot
template = '\n{% block preamble %}\n<style>\n    .grid {\n        display: inline-grid;\n        grid-template-columns: auto auto auto auto;\n        grid-gap: 10px;\n        padding: 10px;\n        background-color: black;\n    }\n</style>\n{% endblock %}\n{% block contents %}\n<div class="grid">\n  {% for root in roots %}\n    {{ embed(root) }}\n  {% endfor %}\n</div>\n{% endblock %}\n'
axes = ['a', 'b', 'l', 'r', 'al', 'ar', 'bl', 'br', 'alr', 'blr', 'lab', 'rab', 'ab', 'lr', 'ablr', '']
figures = list(map(make_figure, axes))
save(figures, template=template, title='CSS grid with an extended template')