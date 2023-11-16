import threading
from visidata import vd, UNLOADED, namedlist, vlen, asyncthread, globalCommand, date
from visidata import VisiData, BaseSheet, Sheet, ColumnAttr, VisiDataMetaSheet, JsonLinesSheet, TypedWrapper, AttrDict, Progress, ErrorSheet, CompleteKey, Path
import visidata
vd.option('replay_wait', 0.0, 'time to wait between replayed commands, in seconds', sheettype=None)
vd.theme_option('disp_replay_play', '▶', 'status indicator for active replay')
vd.theme_option('color_status_replay', 'green', 'color of replay status indicator')
nonLogged = 'forget exec-longname undo redo quit\nshow error errors statuses options threads jump\nreplay cancel save-cmdlog macro cmdlog-sheet menu repeat reload-every\ngo- search scroll prev next page start end zoom resize visibility sidebar\nmouse suspend redraw no-op help syscopy sysopen profile toggle'.split()
vd.option('rowkey_prefix', 'キ', 'string prefix for rowkey in the cmdlog', sheettype=None)
vd.option('cmdlog_histfile', '', 'file to autorecord each cmdlog action to', sheettype=None)
vd.activeCommand = UNLOADED
vd._nextCommands = []
CommandLogRow = namedlist('CommandLogRow', 'sheet col row longname input keystrokes comment undofuncs'.split())

@VisiData.api
def queueCommand(vd, longname, input=None, sheet=None, col=None, row=None):
    if False:
        print('Hello World!')
    'Add command to queue of next commands to execute.'
    vd._nextCommands.append(CommandLogRow(longname=longname, input=input, sheet=sheet, col=col, row=row))

@VisiData.api
def open_vd(vd, p):
    if False:
        for i in range(10):
            print('nop')
    return CommandLog(p.name, source=p, precious=True)

@VisiData.api
def open_vdj(vd, p):
    if False:
        for i in range(10):
            print('nop')
    return CommandLogJsonl(p.name, source=p, precious=True)
VisiData.save_vd = VisiData.save_tsv

@VisiData.api
def save_vdj(vd, p, *vsheets):
    if False:
        print('Hello World!')
    with p.open(mode='w', encoding=vsheets[0].options.save_encoding) as fp:
        fp.write('#!vd -p\n')
        for vs in vsheets:
            vs.write_jsonl(fp)

@VisiData.api
def checkVersion(vd, desired_version):
    if False:
        return 10
    if desired_version != visidata.__version_info__:
        vd.fail('version %s required' % desired_version)

@VisiData.api
def fnSuffix(vd, prefix: str):
    if False:
        print('Hello World!')
    i = 0
    fn = prefix + '.vdj'
    while Path(fn).exists():
        i += 1
        fn = f'{prefix}-{i}.vdj'
    return fn

def indexMatch(L, func):
    if False:
        return 10
    'returns the smallest i for which func(L[i]) is true'
    for (i, x) in enumerate(L):
        if func(x):
            return i

def keystr(k):
    if False:
        i = 10
        return i + 15
    return vd.options.rowkey_prefix + ','.join(map(str, k))

@VisiData.api
def isLoggableCommand(vd, longname):
    if False:
        while True:
            i = 10
    for n in nonLogged:
        if longname.startswith(n):
            return False
    return True

def isLoggableSheet(sheet):
    if False:
        return 10
    return sheet is not vd.cmdlog and (not isinstance(sheet, (vd.OptionsSheet, ErrorSheet)))

@Sheet.api
def moveToRow(vs, rowstr):
    if False:
        while True:
            i = 10
    'Move cursor to row given by *rowstr*, which can be either the row number or keystr.'
    rowidx = vs.getRowIndexFromStr(rowstr)
    if rowidx is None:
        return False
    vs.cursorRowIndex = rowidx
    return True

@Sheet.api
def getRowIndexFromStr(vs, rowstr):
    if False:
        print('Hello World!')
    index = indexMatch(vs.rows, lambda r, vs=vs, rowstr=rowstr: keystr(vs.rowkey(r)) == rowstr)
    if index is not None:
        return index
    try:
        return int(rowstr)
    except ValueError:
        return None

@Sheet.api
def moveToCol(vs, col):
    if False:
        print('Hello World!')
    'Move cursor to column given by *col*, which can be either the column number or column name.'
    if isinstance(col, str):
        vcolidx = indexMatch(vs.visibleCols, lambda c, name=col: name == c.name)
    elif isinstance(col, int):
        vcolidx = col
    if vcolidx is None or vcolidx >= vs.nVisibleCols:
        return False
    vs.cursorVisibleColIndex = vcolidx
    return True

@BaseSheet.api
def commandCursor(sheet, execstr):
    if False:
        for i in range(10):
            print('nop')
    'Return (col, row) of cursor suitable for cmdlog replay of execstr.'
    (colname, rowname) = ('', '')
    contains = lambda s, *substrs: any((a in s for a in substrs))
    if contains(execstr, 'cursorTypedValue', 'cursorDisplay', 'cursorValue', 'cursorCell', 'cursorRow') and sheet.nRows > 0:
        k = sheet.rowkey(sheet.cursorRow)
        rowname = keystr(k) if k else sheet.cursorRowIndex
    if contains(execstr, 'cursorTypedValue', 'cursorDisplay', 'cursorValue', 'cursorCell', 'cursorCol', 'cursorVisibleCol', 'ColumnAtCursor'):
        if sheet.cursorCol:
            colname = sheet.cursorCol.name or sheet.visibleCols.index(sheet.cursorCol)
        else:
            colname = None
    return (colname, rowname)

class CommandLogBase:
    """Log of commands for current session."""
    rowtype = 'logged commands'
    precious = False
    _rowtype = CommandLogRow
    columns = [ColumnAttr('sheet'), ColumnAttr('col'), ColumnAttr('row'), ColumnAttr('longname'), ColumnAttr('input'), ColumnAttr('keystrokes'), ColumnAttr('comment'), ColumnAttr('undo', 'undofuncs', type=vlen, width=0)]
    filetype = 'vd'

    def newRow(self, **fields):
        if False:
            while True:
                i = 10
        return self._rowtype(**fields)

    def beforeExecHook(self, sheet, cmd, args, keystrokes):
        if False:
            i = 10
            return i + 15
        if vd.activeCommand:
            self.afterExecSheet(sheet, False, '')
        (colname, rowname, sheetname) = ('', '', None)
        if sheet and (not (cmd.longname.startswith('open-') and (not cmd.longname in ('open-row', 'open-cell')))):
            sheetname = sheet.name
            (colname, rowname) = sheet.commandCursor(cmd.execstr)
            contains = lambda s, *substrs: any((a in s for a in substrs))
            if contains(cmd.execstr, 'pasteFromClipboard'):
                args = vd.sysclipValue().strip()
        comment = vd.currentReplayRow.comment if vd.currentReplayRow else cmd.helpstr
        vd.activeCommand = self.newRow(sheet=sheetname, col=colname, row=str(rowname), keystrokes=keystrokes, input=args, longname=cmd.longname, comment=comment, undofuncs=[])

    def afterExecSheet(self, sheet, escaped, err):
        if False:
            print('Hello World!')
        'Records vd.activeCommand'
        if not vd.activeCommand:
            return
        if err:
            vd.activeCommand[-1] += ' [%s]' % err
        if escaped:
            vd.activeCommand = None
            return
        if not sheet.cmdlog_sheet.rows or vd.isLoggableCommand(vd.activeCommand.longname):
            if isLoggableSheet(sheet):
                self.addRow(vd.activeCommand)
            sheet.cmdlog_sheet.addRow(vd.activeCommand)
            if vd.options.cmdlog_histfile:
                name = date().strftime(vd.options.cmdlog_histfile)
                p = Path(name)
                if not p.is_absolute():
                    p = Path(sheet.options.visidata_dir) / f'{name}.jsonl'
                if not getattr(vd, 'sessionlog', None):
                    vd.sessionlog = vd.loadInternalSheet(CommandLog, p)
                vd.sessionlog.append_tsv_row(vd.activeCommand)
        vd.activeCommand = None

    def openHook(self, vs, src):
        if False:
            return 10
        while isinstance(src, BaseSheet):
            src = src.source
        r = self.newRow(keystrokes='o', input=str(src), longname='open-file')
        vs.cmdlog_sheet.addRow(r)
        self.addRow(r)

class CommandLog(CommandLogBase, VisiDataMetaSheet):
    pass

class CommandLogJsonl(CommandLogBase, JsonLinesSheet):
    filetype = 'vdj'

    def newRow(self, **fields):
        if False:
            while True:
                i = 10
        return AttrDict(JsonLinesSheet.newRow(self, **fields))

    def iterload(self):
        if False:
            while True:
                i = 10
        for r in JsonLinesSheet.iterload(self):
            if isinstance(r, TypedWrapper):
                yield r
            else:
                yield AttrDict(r)
vd.paused = False
vd.currentReplay = None
vd.currentReplayRow = None

@VisiData.api
def replay_cancel(vd):
    if False:
        return 10
    vd.currentReplayRow = None
    vd.currentReplay = None
    vd._nextCommands.clear()

@VisiData.api
def moveToReplayContext(vd, r, vs):
    if False:
        for i in range(10):
            print('nop')
    'set the sheet/row/col to the values in the replay row'
    vs.ensureLoaded()
    vd.sync()
    vd.clearCaches()
    if r.row not in [None, '']:
        vs.moveToRow(r.row) or vd.error(f'no "{r.row}" row on {vs}')
    if r.col not in [None, '']:
        vs.moveToCol(r.col) or vd.error(f'no "{r.col}" column on {vs}')

@VisiData.api
def replayOne(vd, r):
    if False:
        return 10
    'Replay the command in one given row.'
    vd.currentReplayRow = r
    longname = getattr(r, 'longname', None)
    if r.sheet and longname not in ['set-option', 'unset-option']:
        vs = vd.getSheet(r.sheet) or vd.error('no sheet named %s' % r.sheet)
    else:
        vs = None
    if longname in ['set-option', 'unset-option']:
        try:
            context = vs if r.sheet and vs else vd
            option_scope = r.sheet or r.col or 'global'
            if option_scope == 'override':
                option_scope = 'global'
            if longname == 'set-option':
                context.options.set(r.row, r.input, option_scope)
            else:
                context.options.unset(r.row, option_scope)
            escaped = False
        except Exception as e:
            vd.exceptionCaught(e)
            escaped = True
    else:
        vs = vs or vd.activeSheet
        if vs:
            vd.push(vs)
        else:
            vs = vd.cmdlog
        try:
            vd.moveToReplayContext(r, vs)
            if r.comment:
                vd.status(r.comment)
            escaped = vs.execCommand(longname if longname else r.keystrokes, keystrokes=r.keystrokes)
        except Exception as e:
            vd.exceptionCaught(e)
            escaped = True
    vd.currentReplayRow = None
    if escaped:
        vd.warning('replay aborted during %s' % (longname or r.keystrokes))
    return escaped

@VisiData.api
class DisableAsync:

    def __enter__(self):
        if False:
            while True:
                i = 10
        vd.execAsync = lambda func, *args, sheet=None, **kwargs: func(*args, **kwargs)

    def __exit__(self, exc_type, exc_val, tb):
        if False:
            i = 10
            return i + 15
        vd.execAsync = lambda *args, vd=vd, **kwargs: visidata.VisiData.execAsync(vd, *args, **kwargs)

@VisiData.api
def replay_sync(vd, cmdlog):
    if False:
        i = 10
        return i + 15
    'Replay all commands in *cmdlog*.'
    with vd.DisableAsync():
        cmdlog.cursorRowIndex = 0
        vd.currentReplay = cmdlog
        with Progress(total=len(cmdlog.rows)) as prog:
            while cmdlog.cursorRowIndex < len(cmdlog.rows):
                if vd.currentReplay is None:
                    vd.status('replay canceled')
                    return
                vd.statuses.clear()
                try:
                    if vd.replayOne(cmdlog.cursorRow):
                        vd.replay_cancel()
                        return True
                except Exception as e:
                    vd.replay_cancel()
                    vd.exceptionCaught(e)
                    vd.status('replay canceled')
                    return True
                cmdlog.cursorRowIndex += 1
                prog.addProgress(1)
                if vd.activeSheet:
                    vd.activeSheet.ensureLoaded()
        vd.status('replay complete')
        vd.currentReplay = None

@VisiData.api
def replay(vd, cmdlog):
    if False:
        while True:
            i = 10
    'Inject commands into live execution with interface.'
    vd.push(cmdlog)
    vd._nextCommands.extend(cmdlog.rows)

@VisiData.api
def getLastArgs(vd):
    if False:
        print('Hello World!')
    'Get user input for the currently playing command.'
    if vd.currentReplayRow:
        return vd.currentReplayRow.input
    return None

@VisiData.api
def setLastArgs(vd, args):
    if False:
        print('Hello World!')
    'Set user input on last command, if not already set.'
    if vd.activeCommand is not None and vd.activeCommand is not UNLOADED:
        if not vd.activeCommand.input:
            vd.activeCommand.input = args

@VisiData.property
def replayStatus(vd):
    if False:
        for i in range(10):
            print('nop')
    if vd._nextCommands:
        return f' | [:status_replay] {len(vd._nextCommands)} {vd.options.disp_replay_play}[:]'
    return ''

@BaseSheet.property
def cmdlog(sheet):
    if False:
        i = 10
        return i + 15
    rows = sheet.cmdlog_sheet.rows
    if isinstance(sheet.source, BaseSheet):
        rows = sheet.source.cmdlog.rows + rows
    return CommandLogJsonl(sheet.name + '_cmdlog', source=sheet, rows=rows)

@BaseSheet.lazy_property
def cmdlog_sheet(sheet):
    if False:
        return 10
    c = CommandLogJsonl(sheet.name + '_cmdlog', source=sheet, rows=[])
    if not isinstance(sheet.source, BaseSheet):
        for r in vd.cmdlog.rows:
            if r.sheet == 'global' and r.longname == 'set-option' or r.longname == 'unset-option':
                c.addRow(r)
    return c

@BaseSheet.property
def shortcut(self):
    if False:
        while True:
            i = 10
    if self._shortcut:
        return self._shortcut
    try:
        return str(vd.allSheets.index(self) + 1)
    except ValueError:
        pass
    try:
        return self.cmdlog_sheet.rows[0].keystrokes
    except Exception:
        pass
    return ''

@VisiData.property
def cmdlog(vd):
    if False:
        i = 10
        return i + 15
    if not vd._cmdlog:
        vd._cmdlog = CommandLogJsonl('cmdlog', rows=[])
        vd._cmdlog.resetCols()
        vd.beforeExecHooks.append(vd._cmdlog.beforeExecHook)
    return vd._cmdlog

@VisiData.property
def modifyCommand(vd):
    if False:
        i = 10
        return i + 15
    if vd.activeCommand is not None and vd.isLoggableCommand(vd.activeCommand.longname):
        return vd.activeCommand
    if not vd.cmdlog.rows:
        return None
    return vd.cmdlog.rows[-1]

@CommandLogJsonl.api
@asyncthread
def repeat_for_n(cmdlog, r, n=1):
    if False:
        for i in range(10):
            print('nop')
    r.sheet = r.row = r.col = ''
    for i in range(n):
        vd.replayOne(r)

@CommandLogJsonl.api
@asyncthread
def repeat_for_selected(cmdlog, r):
    if False:
        i = 10
        return i + 15
    r.sheet = r.row = r.col = ''
    for (idx, r) in enumerate(vd.sheet.rows):
        if vd.sheet.isSelected(r):
            vd.sheet.cursorRowIndex = idx
            vd.replayOne(r)
BaseSheet.init('_shortcut')
globalCommand('gD', 'cmdlog-all', 'vd.push(vd.cmdlog)', 'open global CommandLog for all commands executed in current session')
globalCommand('D', 'cmdlog-sheet', 'vd.push(sheet.cmdlog)', "open current sheet's CommandLog with all other loose ends removed; includes commands from parent sheets")
globalCommand('zD', 'cmdlog-sheet-only', 'vd.push(sheet.cmdlog_sheet)', 'open CommandLog for current sheet with commands from parent sheets removed')
BaseSheet.addCommand('^D', 'save-cmdlog', 'saveSheets(inputPath("save cmdlog to: ", value=fnSuffix(name)), vd.cmdlog)', 'save CommandLog to filename.vdj file')
BaseSheet.bindkey('^N', 'no-op')
BaseSheet.addCommand('^K', 'replay-stop', 'vd.replay_cancel(); vd.warning("replay canceled")', 'cancel current replay')
globalCommand(None, 'show-status', 'status(input("status: "))', 'show given message on status line')
globalCommand('^V', 'show-version', 'status(__version_info__);', 'Show version and copyright information on status line')
globalCommand('z^V', 'check-version', 'checkVersion(input("require version: ", value=__version_info__))', 'check VisiData version against given version')
CommandLog.addCommand('x', 'replay-row', 'vd.replayOne(cursorRow); status("replayed one row")', 'replay command in current row')
CommandLog.addCommand('gx', 'replay-all', 'vd.replay(sheet)', 'replay contents of entire CommandLog')
CommandLogJsonl.addCommand('x', 'replay-row', 'vd.replayOne(cursorRow); status("replayed one row")', 'replay command in current row')
CommandLogJsonl.addCommand('gx', 'replay-all', 'vd.replay(sheet)', 'replay contents of entire CommandLog')
CommandLog.options.json_sort_keys = False
CommandLog.options.encoding = 'utf-8'
CommandLogJsonl.options.json_sort_keys = False
vd.addGlobals(CommandLogBase=CommandLogBase, CommandLogRow=CommandLogRow)
vd.addMenuItems('\n            View > Command log > this sheet > cmdlog-sheet\n    View > Command log > this sheet only > cmdlog-sheet-only\n    View > Command log > all commands > cmdlog-all\n    System > Execute longname > exec-longname\n    Help > Version > show-version\n')