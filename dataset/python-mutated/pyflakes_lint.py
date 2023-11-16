from pyflakes import api as pyflakes_api, messages
from pylsp import hookimpl, lsp
PYFLAKES_ERROR_MESSAGES = (messages.UndefinedName, messages.UndefinedExport, messages.UndefinedLocal, messages.DuplicateArgument, messages.FutureFeatureNotDefined, messages.ReturnOutsideFunction, messages.YieldOutsideFunction, messages.ContinueOutsideLoop, messages.BreakOutsideLoop, messages.TwoStarredExpressions)

@hookimpl
def pylsp_lint(workspace, document):
    if False:
        while True:
            i = 10
    with workspace.report_progress('lint: pyflakes'):
        reporter = PyflakesDiagnosticReport(document.lines)
        pyflakes_api.check(document.source.encode('utf-8'), document.path, reporter=reporter)
        return reporter.diagnostics

class PyflakesDiagnosticReport:

    def __init__(self, lines):
        if False:
            for i in range(10):
                print('nop')
        self.lines = lines
        self.diagnostics = []

    def unexpectedError(self, _filename, msg):
        if False:
            i = 10
            return i + 15
        err_range = {'start': {'line': 0, 'character': 0}, 'end': {'line': 0, 'character': 0}}
        self.diagnostics.append({'source': 'pyflakes', 'range': err_range, 'message': msg, 'severity': lsp.DiagnosticSeverity.Error})

    def syntaxError(self, _filename, msg, lineno, offset, text):
        if False:
            i = 10
            return i + 15
        lineno = lineno or 1
        offset = offset or 0
        text = text or ''
        err_range = {'start': {'line': lineno - 1, 'character': offset}, 'end': {'line': lineno - 1, 'character': offset + len(text)}}
        self.diagnostics.append({'source': 'pyflakes', 'range': err_range, 'message': msg, 'severity': lsp.DiagnosticSeverity.Error})

    def flake(self, message):
        if False:
            print('Hello World!')
        'Get message like <filename>:<lineno>: <msg>'
        err_range = {'start': {'line': message.lineno - 1, 'character': message.col}, 'end': {'line': message.lineno - 1, 'character': len(self.lines[message.lineno - 1])}}
        severity = lsp.DiagnosticSeverity.Warning
        for message_type in PYFLAKES_ERROR_MESSAGES:
            if isinstance(message, message_type):
                severity = lsp.DiagnosticSeverity.Error
                break
        self.diagnostics.append({'source': 'pyflakes', 'range': err_range, 'message': message.message % message.message_args, 'severity': severity})