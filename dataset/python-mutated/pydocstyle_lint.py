import contextlib
import logging
import os
import re
import sys
import pydocstyle
from pylsp import hookimpl, lsp
log = logging.getLogger(__name__)
pydocstyle_logger = logging.getLogger(pydocstyle.utils.__name__)
pydocstyle_logger.setLevel(logging.INFO)
DEFAULT_MATCH_RE = pydocstyle.config.ConfigurationParser.DEFAULT_MATCH_RE
DEFAULT_MATCH_DIR_RE = pydocstyle.config.ConfigurationParser.DEFAULT_MATCH_DIR_RE

@hookimpl
def pylsp_settings():
    if False:
        while True:
            i = 10
    return {'plugins': {'pydocstyle': {'enabled': False}}}

@hookimpl
def pylsp_lint(config, workspace, document):
    if False:
        i = 10
        return i + 15
    with workspace.report_progress('lint: pydocstyle'):
        settings = config.plugin_settings('pydocstyle', document_path=document.path)
        log.debug('Got pydocstyle settings: %s', settings)
        filename_match_re = re.compile(settings.get('match', DEFAULT_MATCH_RE) + '$')
        if not filename_match_re.match(os.path.basename(document.path)):
            return []
        dir_match_re = re.compile(settings.get('matchDir', DEFAULT_MATCH_DIR_RE) + '$')
        if not dir_match_re.match(os.path.basename(os.path.dirname(document.path))):
            return []
        args = [document.path]
        if settings.get('convention'):
            args.append('--convention=' + settings['convention'])
            if settings.get('addSelect'):
                args.append('--add-select=' + ','.join(settings['addSelect']))
            if settings.get('addIgnore'):
                args.append('--add-ignore=' + ','.join(settings['addIgnore']))
        elif settings.get('select'):
            args.append('--select=' + ','.join(settings['select']))
        elif settings.get('ignore'):
            args.append('--ignore=' + ','.join(settings['ignore']))
        log.info('Using pydocstyle args: %s', args)
        conf = pydocstyle.config.ConfigurationParser()
        with _patch_sys_argv(args):
            conf.parse()
        diags = []
        for (filename, checked_codes, ignore_decorators, property_decorators, ignore_self_only_init) in conf.get_files_to_check():
            errors = pydocstyle.checker.ConventionChecker().check_source(document.source, filename, ignore_decorators=ignore_decorators, property_decorators=property_decorators, ignore_self_only_init=ignore_self_only_init)
            try:
                for error in errors:
                    if error.code not in checked_codes:
                        continue
                    diags.append(_parse_diagnostic(document, error))
            except pydocstyle.parser.ParseError:
                pass
        log.debug('Got pydocstyle errors: %s', diags)
        return diags

def _parse_diagnostic(document, error):
    if False:
        while True:
            i = 10
    lineno = error.definition.start - 1
    line = document.lines[0] if document.lines else ''
    start_character = len(line) - len(line.lstrip())
    end_character = len(line)
    return {'source': 'pydocstyle', 'code': error.code, 'message': error.message, 'severity': lsp.DiagnosticSeverity.Warning, 'range': {'start': {'line': lineno, 'character': start_character}, 'end': {'line': lineno, 'character': end_character}}}

@contextlib.contextmanager
def _patch_sys_argv(arguments):
    if False:
        while True:
            i = 10
    old_args = sys.argv
    sys.argv = old_args[0:1] + arguments
    try:
        yield
    finally:
        sys.argv = old_args