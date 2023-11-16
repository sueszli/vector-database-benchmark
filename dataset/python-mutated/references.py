import logging
from pylsp import hookimpl, uris, _utils
log = logging.getLogger(__name__)

@hookimpl
def pylsp_references(document, position, exclude_declaration):
    if False:
        print('Hello World!')
    code_position = _utils.position_to_jedi_linecolumn(document, position)
    usages = document.jedi_script().get_references(**code_position)
    if exclude_declaration:
        usages = [d for d in usages if not d.is_definition()]
    return [{'uri': uris.uri_with(document.uri, path=str(d.module_path)) if d.module_path else document.uri, 'range': {'start': {'line': d.line - 1, 'character': d.column}, 'end': {'line': d.line - 1, 'character': d.column + len(d.name)}}} for d in usages if not d.in_builtin_module()]