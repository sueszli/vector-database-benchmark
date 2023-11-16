import unicodedata
import sys
import re
import functools
import textwrap
from visidata import options, drawcache, vd, update_attr, colors, ColorAttr
disp_column_fill = ' '
internal_markup_re = '(\\[[:/][^\\]]*?\\])'
ZERO_WIDTH_CF = set(map(chr, [0, 847, 8203, 8204, 8205, 8206, 8207, 8232, 8233, 8234, 8235, 8236, 8237, 8238, 8288, 8289, 8290, 8291]))

def wcwidth(cc, ambig=1):
    if False:
        while True:
            i = 10
    if cc in ZERO_WIDTH_CF:
        return 1
    eaw = unicodedata.east_asian_width(cc)
    if eaw in 'AN':
        if unicodedata.category(cc) == 'Mn':
            return 1
        else:
            return ambig
    elif eaw in 'WF':
        return 2
    elif not unicodedata.combining(cc):
        return 1
    return 0

def is_vdcode(s: str) -> bool:
    if False:
        return 10
    return s.startswith('[:') and s.endswith(']') or (s.startswith('[/') and s.endswith(']'))

def iterchunks(s, literal=False):
    if False:
        i = 10
        return i + 15
    attrstack = [dict(link='', cattr=ColorAttr())]
    legitopens = 0
    chunks = re.split(internal_markup_re, s)
    for chunk in chunks:
        if not chunk:
            continue
        if not literal and is_vdcode(chunk):
            cattr = attrstack[-1]['cattr']
            link = attrstack[-1]['link']
            if chunk.startswith('[:onclick '):
                attrstack.append(dict(link=chunk[10:-1], cattr=cattr.update(colors.clickable)))
                continue
            elif chunk == '[:]':
                if len(attrstack) > 1:
                    del attrstack[1:]
                    continue
            elif chunk.startswith('[/'):
                if len(attrstack) > 1:
                    attrstack.pop()
                continue
            else:
                newcolor = colors.get_color(chunk[2:-1])
                if newcolor:
                    cattr = update_attr(cattr, newcolor, len(attrstack))
                    attrstack.append(dict(link=link, cattr=cattr))
                    continue
        yield (attrstack[-1], chunk)

@functools.lru_cache(maxsize=100000)
def dispwidth(ss, maxwidth=None, literal=False):
    if False:
        for i in range(10):
            print('nop')
    'Return display width of string, according to unicodedata width and options.disp_ambig_width.'
    disp_ambig_width = options.disp_ambig_width
    w = 0
    for (_, s) in iterchunks(ss, literal=literal):
        for cc in s:
            if cc:
                w += wcwidth(cc, disp_ambig_width)
                if maxwidth and w > maxwidth:
                    return maxwidth
    return w

@functools.lru_cache(maxsize=100000)
def _dispch(c, oddspacech=None, combch=None, modch=None):
    if False:
        i = 10
        return i + 15
    ccat = unicodedata.category(c)
    if ccat in ['Mn', 'Sk', 'Lm']:
        if unicodedata.name(c).startswith('MODIFIER'):
            return (modch, 1)
    elif c != ' ' and ccat in ('Cc', 'Zs', 'Zl', 'Cs'):
        return (oddspacech, 1)
    elif c in ZERO_WIDTH_CF:
        return (combch, 1)
    return (c, dispwidth(c, literal=True))

def iterchars(x):
    if False:
        for i in range(10):
            print('nop')
    if isinstance(x, dict):
        yield from ('{%d}' % len(x))
        for (k, v) in x.items():
            yield ' '
            yield from iterchars(k)
            yield '='
            yield from iterchars(v)
    elif isinstance(x, (list, tuple)):
        yield from ('[%d] ' % len(x))
        for (i, v) in enumerate(x):
            if i != 0:
                yield from '; '
            yield from iterchars(v)
    else:
        yield from str(x)

@functools.lru_cache(maxsize=100000)
def _clipstr(s, dispw, trunch='', oddspacech='', combch='', modch=''):
    if False:
        print('Hello World!')
    "Return clipped string and width in terminal display characters.\n    Note: width may differ from len(s) if East Asian chars are 'fullwidth'."
    if not s:
        return ('', 0)
    if dispw == 1:
        return (s[0], 1)
    w = 0
    ret = ''
    trunchlen = dispwidth(trunch)
    for c in s:
        (newc, chlen) = _dispch(c, oddspacech=oddspacech, combch=combch, modch=modch)
        if not newc:
            newc = c
            chlen = dispwidth(c)
        if dispw and w + chlen > dispw:
            if trunchlen and dispw > trunchlen:
                lastchlen = _dispch(ret[-1])[1]
                if w + trunchlen > dispw:
                    ret = ret[:-1]
                    w -= lastchlen
                ret += trunch
                w += trunchlen
            break
        w += chlen
        ret += newc
    return (ret, w)

@drawcache
def clipstr(s, dispw, truncator=None, oddspace=None):
    if False:
        i = 10
        return i + 15
    if options.visibility:
        return _clipstr(s, dispw, trunch=options.disp_truncator if truncator is None else truncator, oddspacech=options.disp_oddspace if oddspace is None else oddspace, modch='◦', combch='◌')
    else:
        return _clipstr(s, dispw, trunch=options.disp_truncator if truncator is None else truncator, oddspacech=options.disp_oddspace if oddspace is None else oddspace, modch='', combch='')

def clipdraw(scr, y, x, s, attr, w=None, clear=True, literal=False, **kwargs):
    if False:
        print('Hello World!')
    'Draw `s`  at (y,x)-(y,x+w) with curses `attr`, clipping with ellipsis char.\n       If `clear`, clear whole editing area before displaying.\n       If `literal`, do not interpret internal color code markup.\n       Return width drawn (max of w).\n    '
    if not literal:
        chunks = iterchunks(s, literal=literal)
    else:
        chunks = [(dict(link='', cattr=ColorAttr()), s)]
    assert x >= 0, x
    assert y >= 0, y
    return clipdraw_chunks(scr, y, x, chunks, attr, w=w, clear=clear, **kwargs)

def clipdraw_chunks(scr, y, x, chunks, cattr: ColorAttr=ColorAttr(), w=None, clear=True, literal=False, **kwargs):
    if False:
        i = 10
        return i + 15
    'Draw `chunks` (sequence of (color:str, text:str) as from iterchunks) at (y,x)-(y,x+w) with curses `attr`, clipping with ellipsis char.\n       If `clear`, clear whole editing area before displaying.\n       Return width drawn (max of w).\n    '
    if scr:
        (_, windowWidth) = scr.getmaxyx()
    else:
        windowWidth = 80
    totaldispw = 0
    assert isinstance(cattr, ColorAttr), cattr
    origattr = cattr
    origw = w
    clipped = ''
    link = ''
    if w and clear:
        actualw = min(w, windowWidth - x - 1)
        if scr:
            scr.addstr(y, x, disp_column_fill * actualw, cattr.attr)
    try:
        for (colorstate, chunk) in chunks:
            if isinstance(colorstate, str):
                cattr = cattr.update(colors.get_color(colorstate))
            else:
                cattr = origattr.update(colorstate['cattr'])
                link = colorstate['link']
            if not chunk:
                continue
            if origw is None:
                chunkw = dispwidth(chunk, maxwidth=windowWidth - totaldispw)
            else:
                chunkw = origw - totaldispw
            chunkw = min(chunkw, windowWidth - x - 1)
            if chunkw <= 0:
                return totaldispw
            if not scr:
                return totaldispw
            (clipped, dispw) = clipstr(chunk, chunkw, **kwargs)
            scr.addstr(y, x, clipped, cattr.attr)
            if link:
                vd.onMouse(scr, x, y, dispw, 1, BUTTON1_RELEASED=link)
            x += dispw
            totaldispw += dispw
            if chunkw < dispw:
                break
    except Exception as e:
        if vd.options.debug:
            raise
    return totaldispw

def _markdown_to_internal(text):
    if False:
        print('Hello World!')
    'Return markdown-formatted `text` converted to internal formatting (like `[:color]text[/]`).'
    text = re.sub('`(.*?)`', '[:code]\\1[/]', text)
    text = re.sub('\\*\\*(.*?)\\*\\*', '[:bold]\\1[/]', text)
    text = re.sub('\\*(.*?)\\*', '[:italic]\\1[/]', text)
    text = re.sub('\\b_(.*?)_\\b', '[:underline]\\1[/]', text)
    return text

def wraptext(text, width=80, indent=''):
    if False:
        return 10
    '\n    Word-wrap `text` and yield (formatted_line, textonly_line) for each line of at most `width` characters.\n    Formatting like `[:color]text[/]` is ignored for purposes of computing width, and not included in `textonly_line`.\n    '
    import re
    for line in text.splitlines():
        if not line:
            yield ('', '')
            continue
        line = _markdown_to_internal(line)
        chunks = re.split(internal_markup_re, line)
        textchunks = [x for x in chunks if not is_vdcode(x)]
        for (linenum, textline) in enumerate(textwrap.wrap(''.join(textchunks), width=width, drop_whitespace=False)):
            txt = textline
            r = ''
            while chunks:
                c = chunks[0]
                if len(c) > len(txt):
                    r += txt
                    chunks[0] = c[len(txt):]
                    break
                if len(chunks) == 1:
                    r += chunks.pop(0)
                else:
                    chunks.pop(0)
                    r += txt[:len(c)] + chunks.pop(0)
                txt = txt[len(c):]
            if linenum > 0:
                r = indent + r
            yield (r, textline)
        for c in chunks:
            yield (c, '')

def clipbox(scr, lines, attr, title=''):
    if False:
        while True:
            i = 10
    scr.erase()
    scr.bkgd(attr)
    scr.box()
    (h, w) = scr.getmaxyx()
    for (i, line) in enumerate(lines):
        clipdraw(scr, i + 1, 2, line, attr)
    clipdraw(scr, 0, w - len(title) - 6, f'| {title} |', attr)
vd.addGlobals(clipstr=clipstr, clipdraw=clipdraw, clipdraw_chunks=clipdraw_chunks, clipbox=clipbox, dispwidth=dispwidth, iterchars=iterchars, iterchunks=iterchunks, wraptext=wraptext)