"""Linter pluging for flake8"""
import logging
import os.path
import re
import sys
from pathlib import PurePath
from subprocess import PIPE, Popen
from pylsp import hookimpl, lsp
log = logging.getLogger(__name__)
FIX_IGNORES_RE = re.compile('([^a-zA-Z0-9_,]*;.*(\\W+||$))')
UNNECESSITY_CODES = {'F401', 'F504', 'F522', 'F523', 'F841'}

@hookimpl
def pylsp_settings():
    if False:
        print('Hello World!')
    return {'plugins': {'flake8': {'enabled': False}}}

@hookimpl
def pylsp_lint(workspace, document):
    if False:
        for i in range(10):
            print('nop')
    with workspace.report_progress('lint: flake8'):
        config = workspace._config
        settings = config.plugin_settings('flake8', document_path=document.path)
        log.debug('Got flake8 settings: %s', settings)
        ignores = settings.get('ignore', [])
        per_file_ignores = settings.get('perFileIgnores')
        if per_file_ignores:
            prev_file_pat = None
            for path in per_file_ignores:
                try:
                    (file_pat, errors) = path.split(':')
                    prev_file_pat = file_pat
                except ValueError:
                    if prev_file_pat is None:
                        log.warning('skipping a Per-file-ignore with no file pattern')
                        continue
                    file_pat = prev_file_pat
                    errors = path
                if PurePath(document.path).match(file_pat):
                    ignores.extend(errors.split(','))
        opts = {'config': settings.get('config'), 'exclude': settings.get('exclude'), 'extend-ignore': settings.get('extendIgnore'), 'filename': settings.get('filename'), 'hang-closing': settings.get('hangClosing'), 'ignore': ignores or None, 'max-complexity': settings.get('maxComplexity'), 'max-line-length': settings.get('maxLineLength'), 'indent-size': settings.get('indentSize'), 'select': settings.get('select')}
        if opts.get('config') and (not os.path.isabs(opts.get('config'))):
            opts['config'] = os.path.abspath(os.path.expanduser(os.path.expandvars(opts.get('config'))))
            log.debug('using flake8 with config: %s', opts['config'])
        flake8_executable = settings.get('executable', 'flake8')
        args = build_args(opts)
        source = document.source
        output = run_flake8(flake8_executable, args, document, source)
        return parse_stdout(source, output)

def run_flake8(flake8_executable, args, document, source):
    if False:
        while True:
            i = 10
    'Run flake8 with the provided arguments, logs errors\n    from stderr if any.\n    '
    args = [i if not i.startswith('--ignore=') else FIX_IGNORES_RE.sub('', i) for i in args if i is not None]
    if document.path and document.path.startswith(document._workspace.root_path):
        args.extend(['--stdin-display-name', os.path.relpath(document.path, document._workspace.root_path)])
    if not os.path.isfile(flake8_executable) and os.sep in flake8_executable:
        flake8_executable = os.path.abspath(os.path.expanduser(os.path.expandvars(flake8_executable)))
    log.debug("Calling %s with args: '%s'", flake8_executable, args)
    popen_kwargs = {}
    if (cwd := document._workspace.root_path):
        popen_kwargs['cwd'] = cwd
    try:
        cmd = [flake8_executable]
        cmd.extend(args)
        p = Popen(cmd, stdin=PIPE, stdout=PIPE, stderr=PIPE, **popen_kwargs)
    except IOError:
        log.debug("Can't execute %s. Trying with '%s -m flake8'", flake8_executable, sys.executable)
        cmd = [sys.executable, '-m', 'flake8']
        cmd.extend(args)
        p = Popen(cmd, stdin=PIPE, stdout=PIPE, stderr=PIPE, **popen_kwargs)
    (stdout, stderr) = p.communicate(source.encode())
    if stderr:
        log.error("Error while running flake8 '%s'", stderr.decode())
    return stdout.decode()

def build_args(options):
    if False:
        while True:
            i = 10
    'Build arguments for calling flake8.\n\n    Args:\n        options: dictionary of argument names and their values.\n    '
    args = ['-']
    for (arg_name, arg_val) in options.items():
        if arg_val is None:
            continue
        arg = None
        if isinstance(arg_val, list):
            arg = '--{}={}'.format(arg_name, ','.join(arg_val))
        elif isinstance(arg_val, bool):
            if arg_val:
                arg = '--{}'.format(arg_name)
        else:
            arg = '--{}={}'.format(arg_name, arg_val)
        args.append(arg)
    return args

def parse_stdout(source, stdout):
    if False:
        for i in range(10):
            print('nop')
    "\n    Build a diagnostics from flake8's output, it should extract every result and format\n    it into a dict that looks like this:\n        {\n            'source': 'flake8',\n            'code': code, # 'E501'\n            'range': {\n                'start': {\n                    'line': start_line,\n                    'character': start_column,\n                },\n                'end': {\n                    'line': end_line,\n                    'character': end_column,\n                },\n            },\n            'message': msg,\n            'severity': lsp.DiagnosticSeverity.*,\n        }\n\n    Args:\n        document: The document to be linted.\n        stdout: output from flake8\n    Returns:\n        A list of dictionaries.\n    "
    document_lines = source.splitlines(True)
    diagnostics = []
    lines = stdout.splitlines()
    for raw_line in lines:
        parsed_line = re.match('(.*):(\\d*):(\\d*): (\\w*) (.*)', raw_line)
        if not parsed_line:
            log.debug("Flake8 output parser can't parse line '%s'", raw_line)
            continue
        parsed_line = parsed_line.groups()
        if len(parsed_line) != 5:
            log.debug("Flake8 output parser can't parse line '%s'", raw_line)
            continue
        (_, line, character, code, msg) = parsed_line
        line = int(line) - 1
        character = int(character) - 1
        msg = code + ' ' + msg
        severity = lsp.DiagnosticSeverity.Warning
        if code == 'E999' or code[0] == 'F':
            severity = lsp.DiagnosticSeverity.Error
        diagnostic = {'source': 'flake8', 'code': code, 'range': {'start': {'line': line, 'character': character}, 'end': {'line': line, 'character': len(document_lines[line])}}, 'message': msg, 'severity': severity}
        if code in UNNECESSITY_CODES:
            diagnostic['tags'] = [lsp.DiagnosticTag.Unnecessary]
        diagnostics.append(diagnostic)
    return diagnostics