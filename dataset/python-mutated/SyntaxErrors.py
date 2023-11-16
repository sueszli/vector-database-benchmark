""" Handling of syntax errors.

Format SyntaxError/IndentationError exception for output, as well as
raise it for the given source code reference.
"""
from nuitka.PythonVersions import python_version

def formatOutput(e):
    if False:
        i = 10
        return i + 15
    if len(e.args) > 1:
        if len(e.args[1]) == 4:
            (reason, (filename, lineno, colno, message)) = e.args
        else:
            (reason, (filename, lineno, colno, message, _lineno2, _colno2)) = e.args
        if message is None and colno is not None:
            colno = None
        if lineno is not None and lineno == 0:
            lineno = 1
    else:
        (reason,) = e.args
        filename = None
        lineno = None
        colno = None
        message = None
    if hasattr(e, 'msg'):
        reason = e.msg
    if colno is not None and (not e.__class__ is IndentationError or python_version < 912):
        colno = colno - len(message) + len(message.lstrip())
        return '  File "%s", line %d\n    %s\n    %s^\n%s: %s' % (filename, lineno, message.strip(), ' ' * (colno - 1) if colno is not None else '', e.__class__.__name__, reason)
    elif message is not None:
        return '  File "%s", line %d\n    %s\n%s: %s' % (filename, lineno, message.strip(), e.__class__.__name__, reason)
    elif filename is not None:
        return '  File "%s", line %s\n%s: %s' % (filename, lineno, e.__class__.__name__, reason)
    else:
        return '%s: %s' % (e.__class__.__name__, reason)

def raiseSyntaxError(reason, source_ref, display_file=True, display_line=True):
    if False:
        for i in range(10):
            print('nop')
    col_offset = source_ref.getColumnNumber()

    def readSource():
        if False:
            i = 10
            return i + 15
        from .SourceHandling import readSourceLine
        return readSourceLine(source_ref)
    if display_file and display_line:
        source_line = readSource()
        raise SyntaxError(reason, (source_ref.getFilename(), source_ref.getLineNumber(), col_offset, source_line))
    if source_ref is not None:
        if display_line:
            source_line = readSource()
        else:
            source_line = None
        exc = SyntaxError(reason, (source_ref.getFilename(), source_ref.getLineNumber(), col_offset, source_line))
    exc = SyntaxError(reason, (None, None, None, None))
    exc.generated_by_nuitka = True
    raise exc