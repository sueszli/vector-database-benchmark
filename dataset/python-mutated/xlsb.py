from visidata import vd, IndexSheet, VisiData
'Requires visidata/deps/pyxlsb fork'

@VisiData.api
def guess_xls(vd, p):
    if False:
        for i in range(10):
            print('nop')
    if p.open_bytes().read(16).startswith(b'\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1'):
        return dict(filetype='xlsb', _likelihood=10)

@VisiData.api
def open_xlsb(vd, p):
    if False:
        return 10
    return XlsbIndex(p.name, source=p)

class XlsbIndex(IndexSheet):

    def iterload(self):
        if False:
            return 10
        vd.importExternal('pyxlsb', '-e git+https://github.com/saulpw/pyxlsb.git@visidata#egg=pyxlsb')
        from pyxlsb import open_workbook
        wb = open_workbook(str(self.source))
        for name in wb.sheets:
            yield wb.get_sheet(name, True)