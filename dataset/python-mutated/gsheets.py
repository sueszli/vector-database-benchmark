import re
from visidata import vd, VisiData, Sheet, IndexSheet, SequenceSheet, ColumnItem, Path, AttrDict, ColumnAttr, asyncthread
SPREADSHEET_FIELDS = 'properties sheets namedRanges spreadsheetUrl developerMetadata dataSources dataSourceSchedules'.split()
SHEET_FIELDS = 'merges conditionalFormats filterViews protectedRanges basicFilter charts bandedRanges developerMetadata rowGroups columnGroups slicers'.split()

@VisiData.api
def open_gsheets(vd, p):
    if False:
        i = 10
        return i + 15
    m = re.search('([A-z0-9_]{44})', p.given)
    if m:
        return GSheetsIndex(p.name, source=m.groups()[0])
vd.open_g = vd.open_gsheets

@VisiData.lazy_property
def google_discovery(self):
    if False:
        while True:
            i = 10
    googleapiclient = vd.importExternal('googleapiclient', 'google-api-python-client')
    from googleapiclient import discovery
    return discovery

@VisiData.cached_property
def _gsheets(vd):
    if False:
        return 10
    return vd.google_discovery.build('sheets', 'v4', credentials=vd.google_auth('spreadsheets.readonly')).spreadsheets()

@VisiData.cached_property
def _gsheets_rw(vd):
    if False:
        print('Hello World!')
    return vd.google_discovery.build('sheets', 'v4', credentials=vd.google_auth('spreadsheets')).spreadsheets()

class GSheetsIndex(Sheet):
    columns = [ColumnAttr('title', 'properties.title'), ColumnAttr('type', 'properties.sheetType', width=0), ColumnAttr('nRows', 'properties.gridProperties.rowCount', type=int), ColumnAttr('nCols', 'properties.gridProperties.columnCount', type=int)]

    def iterload(self):
        if False:
            for i in range(10):
                print('nop')
        googlesheet = vd._gsheets.get(spreadsheetId=self.source, fields=','.join(SPREADSHEET_FIELDS)).execute()
        vd.status(googlesheet['properties']['title'])
        for gsheet in googlesheet['sheets']:
            yield AttrDict(gsheet)

    def openRow(self, r):
        if False:
            while True:
                i = 10
        return GSheet(r.properties.title, source=self.source)

class GSheet(SequenceSheet):
    """.source is gsheet id; .name is sheet name"""

    def iterload(self):
        if False:
            print('Hello World!')
        result = vd._gsheets.values().get(spreadsheetId=self.source, range=self.name).execute()
        yield from result.get('values', [])

@VisiData.api
def save_gsheets(vd, p, *sheets):
    if False:
        return 10
    gsheet = vd._gsheets_rw.create(body={'properties': {'title': p.name}, 'sheets': list(({'properties': {'title': vs.name}} for vs in sheets))}, fields='spreadsheetId').execute()
    gsheetId = gsheet.get('spreadsheetId')
    vd.status(f'https://docs.google.com/spreadsheets/d/{gsheetId}/')
    for vs in sheets:
        rows = [list((c.name for c in vs.visibleCols))]
        rows += list((list((val for (col, val) in row.items())) for row in vs.iterdispvals(*vs.visibleCols, format=True)))
        vd._gsheets_rw.values().append(spreadsheetId=gsheetId, valueInputOption='RAW', range=vs.name, body=dict(values=rows)).execute()
vd.save_g = vd.save_gsheets