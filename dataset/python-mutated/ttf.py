from visidata import VisiData, vd, Sheet, Column, Progress, ColumnAttr, ColumnItem, SubColumnItem, InvertedCanvas

@VisiData.api
def open_ttf(vd, p):
    if False:
        print('Hello World!')
    return TTFTablesSheet(p.name, source=p)
vd.open_otf = vd.open_ttf

class TTFTablesSheet(Sheet):
    rowtype = 'font tables'
    columns = [ColumnAttr('cmap'), ColumnAttr('format', type=int), ColumnAttr('language', type=int), ColumnAttr('length', type=int), ColumnAttr('platEncID', type=int), ColumnAttr('platformID', type=int), Column('isSymbol', getter=lambda col, row: row.isSymbol()), Column('isUnicode', getter=lambda col, row: row.isUnicode())]

    def openRow(self, row):
        if False:
            print('Hello World!')
        return TTFGlyphsSheet(self.name + '_glyphs', source=self, sourceRows=[row], ttf=self.ttf)

    def iterload(self):
        if False:
            while True:
                i = 10
        fontTools = vd.importExternal('fontTools.ttLib', 'fonttools')
        self.ttf = fontTools.TTFont(str(self.source), 0, allowVID=0, ignoreDecompileErrors=True, fontNumber=-1)
        for cmap in self.ttf['cmap'].tables:
            yield cmap

class TTFGlyphsSheet(Sheet):
    rowtype = 'glyphs'
    columns = [ColumnItem('codepoint', 0, type=int, fmtstr='%0X'), ColumnItem('glyphid', 1), SubColumnItem(2, ColumnAttr('height', type=int)), SubColumnItem(2, ColumnAttr('width', type=int)), SubColumnItem(2, ColumnAttr('lsb')), SubColumnItem(2, ColumnAttr('tsb'))]

    def openRow(self, row):
        if False:
            for i in range(10):
                print('nop')
        return makePen(self.name + '_' + row[1], source=row[2], glyphSet=self.ttf.getGlyphSet())

    def iterload(self):
        if False:
            return 10
        glyphs = self.ttf.getGlyphSet()
        for cmap in self.sourceRows:
            for (codepoint, glyphid) in Progress(cmap.cmap.items(), total=len(cmap.cmap)):
                yield (codepoint, glyphid, glyphs[glyphid])

def makePen(*args, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    fontTools = vd.importExternal('fontTools', 'fonttools')
    from fontTools.pens.basePen import BasePen

    class GlyphPen(InvertedCanvas, BasePen):
        aspectRatio = 1.0

        def __init__(self, name, **kwargs):
            if False:
                return 10
            super().__init__(name, **kwargs)
            self.path_firstxy = None
            self.lastxy = None
            self.attr = self.plotColor(('glyph',))

        def _moveTo(self, xy):
            if False:
                print('Hello World!')
            self.lastxy = xy
            if self.path_firstxy is None:
                self.path_firstxy = xy

        def _lineTo(self, xy):
            if False:
                print('Hello World!')
            (x1, y1) = self.lastxy
            (x2, y2) = xy
            self.line(x1, y1, x2, y2, self.attr)
            self._moveTo(xy)

        def _closePath(self):
            if False:
                print('Hello World!')
            if self.path_firstxy:
                if self.path_firstxy != self.lastxy:
                    self._lineTo(self.path_firstxy)
                self.path_firstxy = None
            self.lastxy = None

        def _endPath(self):
            if False:
                i = 10
                return i + 15
            self.path_firstxy = None
            self.lastxy = None

        def _curveToOne(self, xy1, xy2, xy3):
            if False:
                for i in range(10):
                    print('nop')
            vd.error('NotImplemented')

        def _qCurveToOne(self, xy1, xy2):
            if False:
                while True:
                    i = 10
            self.qcurve([self.lastxy, xy1, xy2], self.attr)
            self._moveTo(xy2)

        def reload(self):
            if False:
                for i in range(10):
                    print('nop')
            self.reset()
            self.source.draw(self)
            self.refresh()
    return GlyphPen(*args, **kwargs)