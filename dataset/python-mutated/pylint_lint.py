"""Linter plugin for pylint."""
import collections
import logging
import sys
import re
from subprocess import Popen, PIPE
import os
import shlex
from pylsp import hookimpl, lsp
try:
    import ujson as json
except Exception:
    import json
log = logging.getLogger(__name__)
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = 'hide'
DEPRECATION_CODES = {'W0402', 'W1505', 'W1511', 'W1512', 'W1513'}
UNNECESSITY_CODES = {'W0611', 'W0612', 'W0613', 'W0614', 'W1304'}

class PylintLinter:
    last_diags = collections.defaultdict(list)

    @classmethod
    def lint(cls, document, is_saved, flags=''):
        if False:
            while True:
                i = 10
        "Plugin interface to pylsp linter.\n\n        Args:\n            document: The document to be linted.\n            is_saved: Whether or not the file has been saved to disk.\n            flags: Additional flags to pass to pylint. Not exposed to\n                pylsp_lint, but used for testing.\n\n        Returns:\n            A list of dicts with the following format:\n\n                {\n                    'source': 'pylint',\n                    'range': {\n                        'start': {\n                            'line': start_line,\n                            'character': start_column,\n                        },\n                        'end': {\n                            'line': end_line,\n                            'character': end_column,\n                        },\n                    }\n                    'message': msg,\n                    'severity': lsp.DiagnosticSeverity.*,\n                }\n        "
        if not is_saved:
            return cls.last_diags[document.path]
        cmd = [sys.executable, '-c', 'import sys; from pylint.lint import Run; Run(sys.argv[1:])', '-f', 'json', document.path] + (shlex.split(str(flags)) if flags else [])
        log.debug("Calling pylint with '%s'", ' '.join(cmd))
        cwd = document._workspace.root_path
        if not cwd:
            cwd = os.path.dirname(__file__)
        with Popen(cmd, stdout=PIPE, stderr=PIPE, cwd=cwd, universal_newlines=True) as process:
            (json_out, err) = process.communicate()
        if err != '':
            log.error("Error calling pylint: '%s'", err)
        if not json_out.strip():
            cls.last_diags[document.path] = []
            return []
        diagnostics = []
        for diag in json.loads(json_out):
            line = diag['line'] - 1
            err_range = {'start': {'line': line, 'character': diag['column']}, 'end': {'line': line, 'character': len(document.lines[line]) if document.lines else 0}}
            if diag['type'] == 'convention':
                severity = lsp.DiagnosticSeverity.Information
            elif diag['type'] == 'information':
                severity = lsp.DiagnosticSeverity.Information
            elif diag['type'] == 'error':
                severity = lsp.DiagnosticSeverity.Error
            elif diag['type'] == 'fatal':
                severity = lsp.DiagnosticSeverity.Error
            elif diag['type'] == 'refactor':
                severity = lsp.DiagnosticSeverity.Hint
            elif diag['type'] == 'warning':
                severity = lsp.DiagnosticSeverity.Warning
            code = diag['message-id']
            diagnostic = {'source': 'pylint', 'range': err_range, 'message': '[{}] {}'.format(diag['symbol'], diag['message']), 'severity': severity, 'code': code}
            if code in UNNECESSITY_CODES:
                diagnostic['tags'] = [lsp.DiagnosticTag.Unnecessary]
            if code in DEPRECATION_CODES:
                diagnostic['tags'] = [lsp.DiagnosticTag.Deprecated]
            diagnostics.append(diagnostic)
        cls.last_diags[document.path] = diagnostics
        return diagnostics

def _build_pylint_flags(settings):
    if False:
        while True:
            i = 10
    'Build arguments for calling pylint.'
    pylint_args = settings.get('args')
    if pylint_args is None:
        return ''
    return ' '.join(pylint_args)

@hookimpl
def pylsp_settings():
    if False:
        for i in range(10):
            print('nop')
    return {'plugins': {'pylint': {'enabled': False, 'args': [], 'executable': None}}}

@hookimpl
def pylsp_lint(config, workspace, document, is_saved):
    if False:
        while True:
            i = 10
    'Run pylint linter.'
    with workspace.report_progress('lint: pylint'):
        settings = config.plugin_settings('pylint')
        log.debug('Got pylint settings: %s', settings)
        if settings.get('executable') and sys.version_info[0] >= 3:
            flags = build_args_stdio(settings)
            pylint_executable = settings.get('executable', 'pylint')
            return pylint_lint_stdin(pylint_executable, document, flags)
        flags = _build_pylint_flags(settings)
        return PylintLinter.lint(document, is_saved, flags=flags)

def build_args_stdio(settings):
    if False:
        print('Hello World!')
    'Build arguments for calling pylint.\n\n    :param settings: client settings\n    :type settings: dict\n\n    :return: arguments to path to pylint\n    :rtype: list\n    '
    pylint_args = settings.get('args')
    if pylint_args is None:
        return []
    return pylint_args

def pylint_lint_stdin(pylint_executable, document, flags):
    if False:
        i = 10
        return i + 15
    'Run pylint linter from stdin.\n\n    This runs pylint in a subprocess with popen.\n    This allows passing the file from stdin and as a result\n    run pylint on unsaved files. Can slowdown the workflow.\n\n    :param pylint_executable: path to pylint executable\n    :type pylint_executable: string\n    :param document: document to run pylint on\n    :type document: pylsp.workspace.Document\n    :param flags: arguments to path to pylint\n    :type flags: list\n\n    :return: linting diagnostics\n    :rtype: list\n    '
    pylint_result = _run_pylint_stdio(pylint_executable, document, flags)
    return _parse_pylint_stdio_result(document, pylint_result)

def _run_pylint_stdio(pylint_executable, document, flags):
    if False:
        i = 10
        return i + 15
    'Run pylint in popen.\n\n    :param pylint_executable: path to pylint executable\n    :type pylint_executable: string\n    :param document: document to run pylint on\n    :type document: pylsp.workspace.Document\n    :param flags: arguments to path to pylint\n    :type flags: list\n\n    :return: result of calling pylint\n    :rtype: string\n    '
    log.debug("Calling %s with args: '%s'", pylint_executable, flags)
    try:
        cmd = [pylint_executable]
        cmd.extend(flags)
        cmd.extend(['--from-stdin', document.path])
        p = Popen(cmd, stdin=PIPE, stdout=PIPE, stderr=PIPE)
    except IOError:
        log.debug("Can't execute %s. Trying with 'python -m pylint'", pylint_executable)
        cmd = ['python', '-m', 'pylint']
        cmd.extend(flags)
        cmd.extend(['--from-stdin', document.path])
        p = Popen(cmd, stdin=PIPE, stdout=PIPE, stderr=PIPE)
    (stdout, stderr) = p.communicate(document.source.encode())
    if stderr:
        log.error("Error while running pylint '%s'", stderr.decode())
    return stdout.decode()

def _parse_pylint_stdio_result(document, stdout):
    if False:
        i = 10
        return i + 15
    'Parse pylint results.\n\n    :param document: document to run pylint on\n    :type document: pylsp.workspace.Document\n    :param stdout: pylint results to parse\n    :type stdout: string\n\n    :return: linting diagnostics\n    :rtype: list\n    '
    diagnostics = []
    lines = stdout.splitlines()
    for raw_line in lines:
        parsed_line = re.match('(.*):(\\d*):(\\d*): (\\w*): (.*)', raw_line)
        if not parsed_line:
            log.debug("Pylint output parser can't parse line '%s'", raw_line)
            continue
        parsed_line = parsed_line.groups()
        if len(parsed_line) != 5:
            log.debug("Pylint output parser can't parse line '%s'", raw_line)
            continue
        (_, line, character, code, msg) = parsed_line
        line = int(line) - 1
        character = int(character)
        severity_map = {'C': lsp.DiagnosticSeverity.Information, 'E': lsp.DiagnosticSeverity.Error, 'F': lsp.DiagnosticSeverity.Error, 'I': lsp.DiagnosticSeverity.Information, 'R': lsp.DiagnosticSeverity.Hint, 'W': lsp.DiagnosticSeverity.Warning}
        severity = severity_map[code[0]]
        diagnostic = {'source': 'pylint', 'code': code, 'range': {'start': {'line': line, 'character': character}, 'end': {'line': line, 'character': len(document.lines[line]) - 1}}, 'message': msg, 'severity': severity}
        if code in UNNECESSITY_CODES:
            diagnostic['tags'] = [lsp.DiagnosticTag.Unnecessary]
        if code in DEPRECATION_CODES:
            diagnostic['tags'] = [lsp.DiagnosticTag.Deprecated]
        diagnostics.append(diagnostic)
    return diagnostics