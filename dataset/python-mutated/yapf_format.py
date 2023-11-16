import logging
import os
from yapf.yapflib import file_resources, style
from yapf.yapflib.yapf_api import FormatCode
import whatthepatch
from pylsp import hookimpl
from pylsp._utils import get_eol_chars
log = logging.getLogger(__name__)

@hookimpl
def pylsp_format_document(workspace, document, options):
    if False:
        print('Hello World!')
    log.info('Formatting document %s with yapf', document)
    with workspace.report_progress('format: yapf'):
        return _format(document, options=options)

@hookimpl
def pylsp_format_range(document, range, options):
    if False:
        print('Hello World!')
    log.info('Formatting document %s in range %s with yapf', document, range)
    range['start']['character'] = 0
    range['end']['line'] += 1
    range['end']['character'] = 0
    lines = [(range['start']['line'] + 1, range['end']['line'] + 1)]
    return _format(document, lines=lines, options=options)

def get_style_config(document_path, options=None):
    if False:
        print('Hello World!')
    exclude_patterns_from_ignore_file = file_resources.GetExcludePatternsForDir(os.getcwd())
    if file_resources.IsIgnored(document_path, exclude_patterns_from_ignore_file):
        return []
    style_config = file_resources.GetDefaultStyleForDir(os.path.dirname(document_path))
    if options is None:
        return style_config
    style_config = style.CreateStyleFromConfig(style_config)
    use_tabs = style_config['USE_TABS']
    indent_width = style_config['INDENT_WIDTH']
    if options.get('tabSize') is not None:
        indent_width = max(int(options.get('tabSize')), 1)
    if options.get('insertSpaces') is not None:
        use_tabs = not options.get('insertSpaces')
        if use_tabs:
            indent_width = 1
    style_config['USE_TABS'] = use_tabs
    style_config['INDENT_WIDTH'] = indent_width
    style_config['CONTINUATION_INDENT_WIDTH'] = indent_width
    for (style_option, value) in options.items():
        if style_option not in style_config:
            continue
        style_config[style_option] = value
    return style_config

def diff_to_text_edits(diff, eol_chars):
    if False:
        while True:
            i = 10
    text_edits = []
    prev_line_no = -1
    for change in diff.changes:
        if change.old and change.new:
            prev_line_no = change.old - 1
        elif change.new:
            text_edits.append({'range': {'start': {'line': prev_line_no + 1, 'character': 0}, 'end': {'line': prev_line_no + 1, 'character': 0}}, 'newText': change.line + eol_chars})
        elif change.old:
            lsp_line_no = change.old - 1
            text_edits.append({'range': {'start': {'line': lsp_line_no, 'character': 0}, 'end': {'line': lsp_line_no + 1, 'character': 0}}, 'newText': ''})
            prev_line_no = lsp_line_no
    return text_edits

def ensure_eof_new_line(document, eol_chars, text_edits):
    if False:
        print('Hello World!')
    if document.source.endswith(eol_chars):
        return
    lines = document.lines
    last_line_number = len(lines) - 1
    if text_edits and text_edits[-1]['range']['start']['line'] >= last_line_number:
        return
    text_edits.append({'range': {'start': {'line': last_line_number, 'character': 0}, 'end': {'line': last_line_number + 1, 'character': 0}}, 'newText': lines[-1] + eol_chars})

def _format(document, lines=None, options=None):
    if False:
        while True:
            i = 10
    source = document.source
    eol_chars = get_eol_chars(source)
    if eol_chars in ['\r', '\r\n']:
        source = source.replace(eol_chars, '\n')
    else:
        eol_chars = '\n'
    style_config = get_style_config(document_path=document.path, options=options)
    (diff_txt, changed) = FormatCode(source, lines=lines, filename=document.filename, print_diff=True, style_config=style_config)
    if not changed:
        return []
    patch_generator = whatthepatch.parse_patch(diff_txt)
    diff = next(patch_generator)
    patch_generator.close()
    text_edits = diff_to_text_edits(diff=diff, eol_chars=eol_chars)
    ensure_eof_new_line(document=document, eol_chars=eol_chars, text_edits=text_edits)
    return text_edits