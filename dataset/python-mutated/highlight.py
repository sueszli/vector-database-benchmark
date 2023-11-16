import logging
from pylsp import hookimpl, lsp, _utils
log = logging.getLogger(__name__)

@hookimpl
def pylsp_document_highlight(document, position):
    if False:
        while True:
            i = 10
    code_position = _utils.position_to_jedi_linecolumn(document, position)
    usages = document.jedi_script().get_references(**code_position)

    def is_valid(definition):
        if False:
            for i in range(10):
                print('nop')
        return definition.line is not None and definition.column is not None

    def local_to_document(definition):
        if False:
            print('Hello World!')
        return not definition.module_path or str(definition.module_path) == document.path
    return [{'range': {'start': {'line': d.line - 1, 'character': d.column}, 'end': {'line': d.line - 1, 'character': d.column + len(d.name)}}, 'kind': lsp.DocumentHighlightKind.Write if d.is_definition() else lsp.DocumentHighlightKind.Read} for d in usages if is_valid(d) and local_to_document(d)]