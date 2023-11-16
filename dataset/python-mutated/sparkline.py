"""
Generate sparkline column for numeric columns
"""
from visidata import vd, Column, Sheet
__author__ = 'Lucas Messenger @layertwo'
vd.theme_option('disp_sparkline', '▁▂▃▄▅▆▇', 'characters to display sparkline')

def sparkline(*values):
    if False:
        print('Hello World!')
    '\n    From *values generate string sparkline\n    '
    lines = vd.options.disp_sparkline
    values = [v for v in values if isinstance(v, (int, float))]
    mx = max(values)
    mn = min(values)
    w = (mx - mn) / len(lines)
    bounds = [mn + w * i for i in range(len(lines))]
    output = ''
    for val in values:
        for b in bounds:
            if mn == 0 and val == 0:
                output += ' '
                break
            if val < b:
                output += lines[bounds.index(b) - 1]
                break
        else:
            output += max(lines)
    return output

@Sheet.api
def addcol_sparkline(sheet, sourceCols):
    if False:
        for i in range(10):
            print('nop')
    '\n    Add sparkline column\n    '
    c = Column('sparkline', sourceCols=sourceCols, getter=lambda c, r: sparkline(*tuple((c.getTypedValue(r) for c in c.sheet.sourceCols))))
    sheet.addColumn(c)
Sheet.addCommand(None, 'addcol-sparkline', 'addcol_sparkline(numericCols(nonKeyVisibleCols))', 'add sparkline of all numeric columns')