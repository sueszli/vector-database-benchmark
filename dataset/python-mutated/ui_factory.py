"""TensorFlow Debugger (tfdbg) User-Interface Factory."""
import copy
SUPPORTED_UI_TYPES = ['readline']

def get_ui(ui_type, on_ui_exit=None, available_ui_types=None, config=None):
    if False:
        for i in range(10):
            print('nop')
    'Create a `base_ui.BaseUI` subtype.\n\n  This factory method attempts to fallback to other available ui_types on\n  ImportError.\n\n  Args:\n    ui_type: (`str`) requested UI type. Currently supported:\n      ( readline)\n    on_ui_exit: (`Callable`) the callback to be called when the UI exits.\n    available_ui_types: (`None` or `list` of `str`) Manually-set available\n      ui_types.\n    config: An instance of `cli_config.CLIConfig()` carrying user-facing\n      configurations.\n\n  Returns:\n    A `base_ui.BaseUI` subtype object.\n\n  Raises:\n    ValueError: on invalid ui_type or on exhausting or fallback ui_types.\n  '
    if available_ui_types is None:
        available_ui_types = copy.deepcopy(SUPPORTED_UI_TYPES)
    if ui_type and ui_type not in available_ui_types:
        raise ValueError("Invalid ui_type: '%s'" % ui_type)
    try:
        if ui_type == 'readline':
            from tensorflow.python.debug.cli import readline_ui
            return readline_ui.ReadlineUI(on_ui_exit=on_ui_exit, config=config)
    except ImportError:
        available_ui_types.remove(ui_type)
        if not available_ui_types:
            raise ValueError('Exhausted all fallback ui_types.')
        return get_ui(available_ui_types[0], available_ui_types=available_ui_types)