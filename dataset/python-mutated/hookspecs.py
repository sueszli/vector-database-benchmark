from pylsp import hookspec

@hookspec
def pylsp_code_actions(config, workspace, document, range, context):
    if False:
        while True:
            i = 10
    pass

@hookspec
def pylsp_code_lens(config, workspace, document):
    if False:
        while True:
            i = 10
    pass

@hookspec
def pylsp_commands(config, workspace):
    if False:
        return 10
    'The list of command strings supported by the server.\n\n    Returns:\n        List[str]: The supported commands.\n    '

@hookspec
def pylsp_completions(config, workspace, document, position, ignored_names):
    if False:
        print('Hello World!')
    pass

@hookspec(firstresult=True)
def pylsp_completion_item_resolve(config, workspace, document, completion_item):
    if False:
        for i in range(10):
            print('nop')
    pass

@hookspec
def pylsp_definitions(config, workspace, document, position):
    if False:
        return 10
    pass

@hookspec
def pylsp_dispatchers(config, workspace):
    if False:
        i = 10
        return i + 15
    pass

@hookspec
def pylsp_document_did_open(config, workspace, document):
    if False:
        for i in range(10):
            print('nop')
    pass

@hookspec
def pylsp_document_did_save(config, workspace, document):
    if False:
        while True:
            i = 10
    pass

@hookspec
def pylsp_document_highlight(config, workspace, document, position):
    if False:
        while True:
            i = 10
    pass

@hookspec
def pylsp_document_symbols(config, workspace, document):
    if False:
        i = 10
        return i + 15
    pass

@hookspec(firstresult=True)
def pylsp_execute_command(config, workspace, command, arguments):
    if False:
        i = 10
        return i + 15
    pass

@hookspec
def pylsp_experimental_capabilities(config, workspace):
    if False:
        while True:
            i = 10
    pass

@hookspec
def pylsp_folding_range(config, workspace, document):
    if False:
        i = 10
        return i + 15
    pass

@hookspec(firstresult=True)
def pylsp_format_document(config, workspace, document, options):
    if False:
        i = 10
        return i + 15
    pass

@hookspec(firstresult=True)
def pylsp_format_range(config, workspace, document, range, options):
    if False:
        for i in range(10):
            print('nop')
    pass

@hookspec(firstresult=True)
def pylsp_hover(config, workspace, document, position):
    if False:
        while True:
            i = 10
    pass

@hookspec
def pylsp_initialize(config, workspace):
    if False:
        return 10
    pass

@hookspec
def pylsp_initialized():
    if False:
        for i in range(10):
            print('nop')
    pass

@hookspec
def pylsp_lint(config, workspace, document, is_saved):
    if False:
        i = 10
        return i + 15
    pass

@hookspec
def pylsp_references(config, workspace, document, position, exclude_declaration):
    if False:
        while True:
            i = 10
    pass

@hookspec(firstresult=True)
def pylsp_rename(config, workspace, document, position, new_name):
    if False:
        return 10
    pass

@hookspec
def pylsp_settings(config):
    if False:
        print('Hello World!')
    pass

@hookspec(firstresult=True)
def pylsp_signature_help(config, workspace, document, position):
    if False:
        print('Hello World!')
    pass

@hookspec
def pylsp_workspace_configuration_changed(config, workspace):
    if False:
        print('Hello World!')
    pass