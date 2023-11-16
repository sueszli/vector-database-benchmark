import pyqtgraph as pg

def test_legend_item_basics():
    if False:
        print('Hello World!')
    pg.mkQApp()
    legend = pg.LegendItem()
    assert legend.opts['pen'] == pg.mkPen(None)
    assert legend.opts['brush'] == pg.mkBrush(None)
    assert legend.opts['labelTextColor'] is None
    assert legend.opts['labelTextSize'] == '9pt'
    assert legend.opts['offset'] is None
    assert legend.columnCount == 1
    assert legend.rowCount == 1
    assert legend.labelTextColor() is None
    assert legend.labelTextSize() == '9pt'
    assert legend.brush() == pg.mkBrush(None)
    assert legend.pen() == pg.mkPen(None)
    assert legend.sampleType is pg.ItemSample
    brush = pg.mkBrush('b')
    legend.setBrush(brush)
    assert legend.brush() == brush
    assert legend.opts['brush'] == brush
    pen = pg.mkPen('b')
    legend.setPen(pen)
    assert legend.pen() == pen
    assert legend.opts['pen'] == pen
    text_color = pg.mkColor('b')
    legend.setLabelTextColor(text_color)
    assert legend.labelTextColor() == text_color
    assert legend.opts['labelTextColor'] == text_color
    text_size = '12pt'
    legend.setLabelTextSize(text_size)
    assert legend.labelTextSize() == text_size
    assert legend.opts['labelTextSize'] == text_size
    assert len(legend.items) == 0
    plot = pg.PlotDataItem(name='Plot')
    legend.addItem(plot, name='Plot')
    assert len(legend.items) == 1
    scatter = pg.PlotDataItem(name='Scatter')
    legend.addItem(scatter, name='Scatter')
    assert len(legend.items) == 2
    assert legend.columnCount == 1
    assert legend.rowCount == 2
    curve = pg.PlotDataItem(name='Curve')
    legend.addItem(curve, name='Curve')
    assert len(legend.items) == 3
    assert legend.rowCount == 3
    scrabble = pg.PlotDataItem(name='Scrabble')
    legend.addItem(scrabble, name='Scrabble')
    assert len(legend.items) == 4
    assert legend.layout.rowCount() == 4
    assert legend.rowCount == 4
    legend.setColumnCount(2)
    assert legend.columnCount == 2
    assert legend.rowCount == 2
    assert legend.layout.rowCount() == 2
    legend.removeItem(scrabble)
    assert legend.rowCount == 2
    assert legend.layout.rowCount() == 2
    assert scrabble not in legend.items
    assert len(legend.items) == 3
    legend.removeItem(curve)
    assert legend.rowCount == 2
    assert legend.layout.rowCount() == 1
    assert curve not in legend.items
    assert len(legend.items) == 2
    legend.clear()
    assert legend.items == []