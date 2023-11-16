from h2o_wave import ui, Q
import json
import file_utils

def update_file_tree(q: Q, root: str) -> None:
    if False:
        i = 10
        return i + 15
    q.page['meta'].script = ui.inline_script(f'eventBus.emit("folder", {json.dumps(file_utils.get_file_tree(root))})')

def open_file(q: Q, file: str) -> None:
    if False:
        return 10
    q.page['meta'].script = ui.inline_script(f"\neditor.setValue(`{file_utils.read_file(file)}`)\neventBus.emit('activeFile', '{file}')\n")

def clean_editor(q: Q) -> None:
    if False:
        i = 10
        return i + 15
    q.page['meta'].script = ui.inline_script(f'editor.setValue(``)')