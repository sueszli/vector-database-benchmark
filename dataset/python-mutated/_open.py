import os
import os.path
from visidata import VisiData, vd, Path, BaseSheet, TableSheet, TextSheet, SettableColumn
vd.option('filetype', '', 'specify file type', replay=True)

@VisiData.api
def inputFilename(vd, prompt, *args, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    completer = _completeFilename
    if not vd.couldOverwrite():
        completer = None
        v = kwargs.get('value', '')
        if v and Path(v).exists():
            kwargs['value'] = ''
    return vd.input(prompt, *args, type='filename', completer=completer, **kwargs).strip()

@VisiData.api
def inputPath(vd, *args, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    return Path(vd.inputFilename(*args, **kwargs))

def _completeFilename(val, state):
    if False:
        while True:
            i = 10
    i = val.rfind('/')
    if i < 0:
        base = ''
        partial = val
    elif i == 0:
        base = '/'
        partial = val[1:]
    else:
        base = val[:i]
        partial = val[i + 1:]
    files = []
    for f in os.listdir(Path(base or '.')):
        if f.startswith(partial):
            files.append(os.path.join(base, f))
    files.sort()
    return files[state % len(files)]

@VisiData.api
def guessFiletype(vd, p, *args, funcprefix='guess_'):
    if False:
        print('Hello World!')
    'Call all vd.guess_<filetype>(p) functions and return best candidate sheet based on file contents.'
    guessfuncs = [getattr(vd, x) for x in dir(vd) if x.startswith(funcprefix)]
    filetypes = []
    for f in guessfuncs:
        try:
            filetype = f(p, *args)
            if filetype:
                filetype['_guesser'] = f.__name__
                filetypes.append(filetype)
        except FileNotFoundError:
            pass
        except Exception as e:
            vd.debug(f'{f.__name__}: {e}')
    if filetypes:
        return sorted(filetypes, key=lambda r: -r.get('_likelihood', 1))[0]
    return {}

@VisiData.api
def guess_extension(vd, path):
    if False:
        for i in range(10):
            print('nop')
    ext = path.suffix[1:].lower()
    openfunc = getattr(vd, f'open_{ext}', vd.getGlobals().get(f'open_{ext}'))
    if openfunc:
        return dict(filetype=ext, _likelihood=3)

@VisiData.api
def openPath(vd, p, filetype=None, create=False):
    if False:
        i = 10
        return i + 15
    'Call ``open_<filetype>(p)`` or ``openurl_<p.scheme>(p, filetype)``.  Return constructed but unloaded sheet of appropriate type.\n    If True, *create* will return a new, blank **Sheet** if file does not exist.'
    if p.scheme and (not p.has_fp()):
        schemes = p.scheme.split('+')
        openfuncname = 'openurl_' + schemes[-1]
        openfunc = getattr(vd, openfuncname, None) or vd.getGlobals().get(openfuncname, None)
        if not openfunc:
            vd.fail(f'no loader for url scheme: {p.scheme}')
        return openfunc(p, filetype=filetype)
    if not p.exists() and (not create):
        return None
    if not filetype:
        filetype = p.ext or vd.options.filetype
    filetype = filetype.lower()
    if not p.exists():
        newfunc = getattr(vd, 'new_' + filetype, vd.getGlobals().get('new_' + filetype))
        if not newfunc:
            vd.warning('%s does not exist, creating new sheet' % p)
            return vd.newSheet(p.name, 1, source=p)
        vd.status('creating blank %s' % p.given)
        return newfunc(p)
    if p.is_fifo():
        p = Path(p.given, fp=p.open(mode='rb'))
    openfuncname = 'open_' + filetype
    openfunc = getattr(vd, openfuncname, vd.getGlobals().get(openfuncname))
    if not openfunc:
        opts = vd.guessFiletype(p)
        if opts and 'filetype' in opts:
            filetype = opts['filetype']
            openfuncname = 'open_' + filetype
            openfunc = getattr(vd, openfuncname, vd.getGlobals().get(openfuncname))
            if not openfunc:
                vd.error(f'guessed {filetype} but no {openfuncname}')
            vs = openfunc(p)
            for (k, v) in opts.items():
                if k != 'filetype' and (not k.startswith('_')):
                    setattr(vs.options, k, v)
            vd.warning('guessed "%s" filetype based on contents' % opts['filetype'])
            return vs
        vd.warning('unknown "%s" filetype' % filetype)
        filetype = 'txt'
        openfunc = vd.open_txt
    vd.status('opening %s as %s' % (p.given, filetype))
    return openfunc(p)

@VisiData.api
def openSource(vd, p, filetype=None, create=False, **kwargs):
    if False:
        print('Hello World!')
    'Return unloaded sheet object for *p* opened as the given *filetype* and with *kwargs* as option overrides. *p* can be a Path or a string (filename, url, or "-" for stdin).\n    when true, *create* will return a blank sheet, if file does not exist.'
    if isinstance(p, BaseSheet):
        return p
    filetype = filetype or vd.options.getonly('filetype', str(p), '')
    filetype = filetype or vd.options.getonly('filetype', 'global', '')
    vs = None
    if isinstance(p, str):
        if '://' in p:
            vs = vd.openPath(Path(p), filetype=filetype)
        elif p == '-':
            vs = vd.openPath(vd.stdinSource, filetype=filetype)
        else:
            vs = vd.openPath(Path(p), filetype=filetype, create=create)
    else:
        vs = vd.openPath(p, filetype=filetype, create=create)
    for (optname, optval) in kwargs.items():
        vs.options[optname] = optval
    return vs

@VisiData.api
def open_txt(vd, p):
    if False:
        while True:
            i = 10
    'Create sheet from `.txt` file at Path `p`, checking whether it is TSV.'
    if p.exists():
        with p.open(encoding=vd.options.encoding) as fp:
            delimiter = vd.options.delimiter
            try:
                if delimiter and delimiter in next(fp):
                    return vd.open_tsv(p)
            except StopIteration:
                return TableSheet(p.name, columns=[SettableColumn()], source=p)
    return TextSheet(p.name, source=p)

@VisiData.api
def loadInternalSheet(vd, cls, p, **kwargs):
    if False:
        print('Hello World!')
    'Load internal sheet of given class.'
    vs = cls(p.name, source=p, **kwargs)
    vd.options._set('encoding', 'utf8', vs)
    if p.exists():
        vs.reload.__wrapped__(vs)
    return vs
BaseSheet.addCommand('o', 'open-file', 'vd.push(openSource(inputFilename("open: "), create=True))', 'Open file or URL')
TableSheet.addCommand('zo', 'open-cell-file', 'vd.push(openSource(cursorDisplay) or fail(f"file {cursorDisplay} does not exist"))', 'Open file or URL from path in current cell')
BaseSheet.addCommand('gU', 'undo-last-quit', 'push(allSheets[-1])', 'reopen most recently closed sheet')
vd.addMenuItems('\n    File > Open > input file/url > open-file\n    File > Reopen last closed > undo-last-quit\n')