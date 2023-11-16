import sys
from . import EpsImagePlugin

class PSDraw:
    """
    Sets up printing to the given file. If ``fp`` is omitted,
    ``sys.stdout.buffer`` or ``sys.stdout`` is assumed.
    """

    def __init__(self, fp=None):
        if False:
            while True:
                i = 10
        if not fp:
            try:
                fp = sys.stdout.buffer
            except AttributeError:
                fp = sys.stdout
        self.fp = fp

    def begin_document(self, id=None):
        if False:
            print('Hello World!')
        'Set up printing of a document. (Write PostScript DSC header.)'
        self.fp.write(b'%!PS-Adobe-3.0\nsave\n/showpage { } def\n%%EndComments\n%%BeginDocument\n')
        self.fp.write(EDROFF_PS)
        self.fp.write(VDI_PS)
        self.fp.write(b'%%EndProlog\n')
        self.isofont = {}

    def end_document(self):
        if False:
            return 10
        'Ends printing. (Write PostScript DSC footer.)'
        self.fp.write(b'%%EndDocument\nrestore showpage\n%%End\n')
        if hasattr(self.fp, 'flush'):
            self.fp.flush()

    def setfont(self, font, size):
        if False:
            for i in range(10):
                print('nop')
        '\n        Selects which font to use.\n\n        :param font: A PostScript font name\n        :param size: Size in points.\n        '
        font = bytes(font, 'UTF-8')
        if font not in self.isofont:
            self.fp.write(b'/PSDraw-%s ISOLatin1Encoding /%s E\n' % (font, font))
            self.isofont[font] = 1
        self.fp.write(b'/F0 %d /PSDraw-%s F\n' % (size, font))

    def line(self, xy0, xy1):
        if False:
            i = 10
            return i + 15
        '\n        Draws a line between the two points. Coordinates are given in\n        PostScript point coordinates (72 points per inch, (0, 0) is the lower\n        left corner of the page).\n        '
        self.fp.write(b'%d %d %d %d Vl\n' % (*xy0, *xy1))

    def rectangle(self, box):
        if False:
            for i in range(10):
                print('nop')
        '\n        Draws a rectangle.\n\n        :param box: A tuple of four integers, specifying left, bottom, width and\n           height.\n        '
        self.fp.write(b'%d %d M 0 %d %d Vr\n' % box)

    def text(self, xy, text):
        if False:
            return 10
        '\n        Draws text at the given position. You must use\n        :py:meth:`~PIL.PSDraw.PSDraw.setfont` before calling this method.\n        '
        text = bytes(text, 'UTF-8')
        text = b'\\('.join(text.split(b'('))
        text = b'\\)'.join(text.split(b')'))
        xy += (text,)
        self.fp.write(b'%d %d M (%s) S\n' % xy)

    def image(self, box, im, dpi=None):
        if False:
            print('Hello World!')
        'Draw a PIL image, centered in the given box.'
        if not dpi:
            if im.mode == '1':
                dpi = 200
            else:
                dpi = 100
        x = im.size[0] * 72 / dpi
        y = im.size[1] * 72 / dpi
        xmax = float(box[2] - box[0])
        ymax = float(box[3] - box[1])
        if x > xmax:
            y = y * xmax / x
            x = xmax
        if y > ymax:
            x = x * ymax / y
            y = ymax
        dx = (xmax - x) / 2 + box[0]
        dy = (ymax - y) / 2 + box[1]
        self.fp.write(b'gsave\n%f %f translate\n' % (dx, dy))
        if (x, y) != im.size:
            sx = x / im.size[0]
            sy = y / im.size[1]
            self.fp.write(b'%f %f scale\n' % (sx, sy))
        EpsImagePlugin._save(im, self.fp, None, 0)
        self.fp.write(b'\ngrestore\n')
EDROFF_PS = b'/S { show } bind def\n/P { moveto show } bind def\n/M { moveto } bind def\n/X { 0 rmoveto } bind def\n/Y { 0 exch rmoveto } bind def\n/E {    findfont\n        dup maxlength dict begin\n        {\n                1 index /FID ne { def } { pop pop } ifelse\n        } forall\n        /Encoding exch def\n        dup /FontName exch def\n        currentdict end definefont pop\n} bind def\n/F {    findfont exch scalefont dup setfont\n        [ exch /setfont cvx ] cvx bind def\n} bind def\n'
VDI_PS = b'/Vm { moveto } bind def\n/Va { newpath arcn stroke } bind def\n/Vl { moveto lineto stroke } bind def\n/Vc { newpath 0 360 arc closepath } bind def\n/Vr {   exch dup 0 rlineto\n        exch dup 0 exch rlineto\n        exch neg 0 rlineto\n        0 exch neg rlineto\n        setgray fill } bind def\n/Tm matrix def\n/Ve {   Tm currentmatrix pop\n        translate scale newpath 0 0 .5 0 360 arc closepath\n        Tm setmatrix\n} bind def\n/Vf { currentgray exch setgray fill setgray } bind def\n'
ERROR_PS = b'/landscape false def\n/errorBUF 200 string def\n/errorNL { currentpoint 10 sub exch pop 72 exch moveto } def\nerrordict begin /handleerror {\n    initmatrix /Courier findfont 10 scalefont setfont\n    newpath 72 720 moveto $error begin /newerror false def\n    (PostScript Error) show errorNL errorNL\n    (Error: ) show\n        /errorname load errorBUF cvs show errorNL errorNL\n    (Command: ) show\n        /command load dup type /stringtype ne { errorBUF cvs } if show\n        errorNL errorNL\n    (VMstatus: ) show\n        vmstatus errorBUF cvs show ( bytes available, ) show\n        errorBUF cvs show ( bytes used at level ) show\n        errorBUF cvs show errorNL errorNL\n    (Operand stargck: ) show errorNL /ostargck load {\n        dup type /stringtype ne { errorBUF cvs } if 72 0 rmoveto show errorNL\n    } forall errorNL\n    (Execution stargck: ) show errorNL /estargck load {\n        dup type /stringtype ne { errorBUF cvs } if 72 0 rmoveto show errorNL\n    } forall\n    end showpage\n} def end\n'