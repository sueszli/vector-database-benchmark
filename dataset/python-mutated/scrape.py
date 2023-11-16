__all__ = ['SelectorColumn', 'soupstr']
import os.path
from urllib.parse import urljoin
import concurrent.futures
import functools
from visidata import vd, VisiData, TableSheet, vdtype, Column, AttrColumn, Progress, date

@VisiData.api
def soup(vd, s):
    if False:
        for i in range(10):
            print('nop')
    bs4 = vd.importExternal('bs4', 'beautifulsoup4')
    from bs4 import BeautifulSoup
    return BeautifulSoup(s, 'html.parser')

@VisiData.api
def open_scrape(vd, p):
    if False:
        i = 10
        return i + 15
    bs4 = vd.importExternal('bs4', 'beautifulsoup4')
    vd.enable_requests_cache()
    if p.is_url():
        return HtmlDocsSheet(p.name, source=p, urls=[p.given])
    else:
        return HtmlElementsSheet(p.name, source=p, elements=None)
VisiData.openhttp_scrape = VisiData.open_scrape

def node_name(node):
    if False:
        while True:
            i = 10
    me = node.name
    class_ = node.attrs.get('class')
    if class_:
        me += '.' + class_[0]
    id_ = node.attrs.get('id')
    if id_:
        me += '#' + id_
    return me

@functools.lru_cache(maxsize=None)
def calc_selector(node):
    if False:
        return 10
    if not node.parent:
        return ''
    psel = calc_selector(node.parent)
    oursel = node_name(node)
    if not psel:
        return oursel
    root = list(node.parents)[-1]
    combinedsel = psel + ' ' + oursel
    if len(root.select(combinedsel)) == len(root.select(oursel)):
        return oursel
    return combinedsel

class HtmlAttrColumn(Column):

    def calcValue(self, row):
        if False:
            return 10
        return row.attrs.get(self.expr)

def prev_header(r):
    if False:
        i = 10
        return i + 15
    hdrtags = ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']
    try:
        i = hdrtags.index(r.name)
        return r.find_previous(hdrtags[:i - 1])
    except Exception:
        return r.find_previous(hdrtags)

class HtmlElementsSheet(TableSheet):
    help = '# HTMLElements\n\nThis is a list of HTML elements from _{sheet.source}_ as parsed by `beautifulsoup4`.\n\nStandard VisiData exploration techniques can be used to find relevant data, which will help determine the proper selector to use.\n\n- `Enter` to dive into children of cursor element (or children of all selected rows with `g Enter`)\n- `go` to batch open links in selected rows on new RequestsSheet, which will fetch each page\n- `~` to use the `soupstr` type to join all the text elements\n'
    rowtype = 'dom nodes'
    columns = [Column('name', getter=lambda c, r: node_name(r)), Column('selector', getter=lambda c, r: calc_selector(r), cache='async', width=0), AttrColumn('string'), Column('depth', cache=True, getter=lambda c, r: list(c.sheet.html_parents(r))), Column('prev_header', getter=lambda c, r: prev_header(r), cache=True), HtmlAttrColumn('href', expr='href')]

    def iterload(self):
        if False:
            while True:
                i = 10
        for el in self.elements or [vd.soup(self.source.read_text())]:
            for x in el.find_all():
                if x.string:
                    yield x

    def html_parents(self, row):
        if False:
            for i in range(10):
                print('nop')
        while row.parent and row.parent is not row:
            yield row.parent
            row = row.parent

    @property
    def rootSource(self):
        if False:
            print('Hello World!')
        return self.rootSheet.source

    def openRows(self, rows):
        if False:
            for i in range(10):
                print('nop')
        realurls = [urljoin(self.rootSource.given, r.attrs.get('href')) for r in rows]
        yield HtmlDocsSheet(self.name, 'scrape', source=self, urls=realurls)

    def openRow(self, row):
        if False:
            i = 10
            return i + 15
        'opening a single row'
        return HtmlElementsSheet('', source=self, elements=[row])

class DocsSelectorColumn(Column):

    def calcValue(self, row):
        if False:
            return 10
        return [x for x in row.soup.select(self.expr)]

class SelectorColumn(Column):

    def calcValue(self, row):
        if False:
            print('Hello World!')
        return [x for x in row.select(self.expr)]

class HtmlDocsSheet(TableSheet):
    help = '# HtmlDocsSheet\n\n- `Enter` to open the current request as list of HTMLElements\n- `;` to add column of elements matching given css selector\n  - this is how to cross-tabulate data from multiple pages\n'
    rowtype = 'requests'
    columns = [AttrColumn('url'), AttrColumn('status_code', type=int), AttrColumn('from_cache'), AttrColumn('fetched_at', 'created_at', type=date, width=0), AttrColumn('expires', type=date), AttrColumn('reason'), AttrColumn('soup.title.string')]

    def iterload(self):
        if False:
            while True:
                i = 10
        requests = vd.importExternal('requests')
        self.colnames = {}
        for url in Progress(self.urls):
            yield requests.get(url)

    def addRow(self, row, index=None):
        if False:
            while True:
                i = 10
        super().addRow(row, index=index)
        try:
            row.soup = vd.soup(row.text)
        except Exception as e:
            vd.exceptionCaught(e)
            row.soup = None

    def openRow(self, row):
        if False:
            for i in range(10):
                print('nop')
        return HtmlElementsSheet(row.url, source=self, elements=[row.soup])

def soupstr(coll):
    if False:
        print('Hello World!')
    return ' '.join((v.string for v in coll))
vdtype(soupstr, 's')

@TableSheet.api
def scrape_urls(sheet, col, rows):
    if False:
        i = 10
        return i + 15
    return HtmlDocsSheet(sheet.name, 'selected_urls', urls=[col.getTypedValue(r) for r in rows])
HtmlElementsSheet.addCommand('~', 'type-soupstr', 'cursorCol.type=soupstr', 'set type of current column to list of html elements')
HtmlElementsSheet.addCommand('go', 'open-rows', 'for vs in openRows(selectedRows): vd.push(vs)', 'open sheet for each selected element')
TableSheet.addCommand('gzo', 'scrape-cells', 'vd.push(scrape_urls(cursorCol, selectedRows))', 'open HTML Documents sheet from selected URLs')
HtmlDocsSheet.addCommand(';', 'addcol-selector', 'sel=input("css selector: ", type="selector"); addColumn(DocsSelectorColumn(sel, expr=sel, cache="async"))', 'add column derived from css selector of current column')
HtmlElementsSheet.addCommand(';', 'addcol-selector', 'sel=input("css selector: ", type="selector"); addColumn(SelectorColumn(sel, expr=sel, cache="async"))', 'add column derived from css selector of current column')
vd.addGlobals({'HtmlDocsSheet': SelectorColumn, 'SelectorColumn': SelectorColumn, 'DocsSelectorColumn': DocsSelectorColumn, 'soupstr': soupstr})
vd.addMenuItem('Data', '+Scrape', 'selected cells', 'scrape-cells')