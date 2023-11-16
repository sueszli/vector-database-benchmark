"""Set options.ntk_key='<api-key>', then use command "addcol-geocode" on a column
with placename, to add a "geocodes" column with a list of potential geocoded
places, and lat/long columns with the actual coords of the first of those."""
__author__ = 'Saul Pwanson <vd+geocode@saul.pw>'
__version__ = '0.1'
import urllib.parse
import json
from visidata import vd, AttrDict, Sheet, Column, ColumnExpr
vd.option('ntk_key', '', 'API Key for nettoolkit.com')

def geocode(addr):
    if False:
        i = 10
        return i + 15
    'Return list of dict of location information given an address query.'
    url = 'https://api.nettoolkit.com/v1/geo/geocodes?address=' + urllib.parse.quote(addr, safe='')
    resp = vd.urlcache(url, headers={'X-NTK-KEY': vd.options.ntk_key})
    return json.loads(resp.read_text())['results']

@Sheet.api
def geocode_col(sheet, vcolidx):
    if False:
        print('Hello World!')
    col = sheet.visibleCols[vcolidx]
    for c in [Column('geocodes', origCol=col, cache='async', getter=lambda c, r: geocode(c.origCol.getDisplayValue(r))), ColumnExpr('lat', origCol=col, cache=False, expr='geocodes[0]["latitude"]'), ColumnExpr('long', origCol=col, cache=False, expr='geocodes[0]["longitude"]')]:
        sheet.addColumn(c, index=vcolidx + 1)
Sheet.addCommand('', 'addcol-geocode', 'sheet.geocode_col(cursorVisibleColIndex)')