"""Path helpers utility functions."""
from tensorflow.python.lib.io import file_io
from tensorflow.python.saved_model import constants
from tensorflow.python.util import compat

def get_or_create_variables_dir(export_dir):
    if False:
        return 10
    "Return variables sub-directory, or create one if it doesn't exist."
    variables_dir = get_variables_dir(export_dir)
    file_io.recursive_create_dir(variables_dir)
    return variables_dir

def get_variables_dir(export_dir):
    if False:
        i = 10
        return i + 15
    'Return variables sub-directory in the SavedModel.'
    return file_io.join(compat.as_text(export_dir), compat.as_text(constants.VARIABLES_DIRECTORY))

def get_variables_path(export_dir):
    if False:
        while True:
            i = 10
    'Return the variables path, used as the prefix for checkpoint files.'
    return file_io.join(compat.as_text(get_variables_dir(export_dir)), compat.as_text(constants.VARIABLES_FILENAME))

def get_or_create_assets_dir(export_dir):
    if False:
        i = 10
        return i + 15
    "Return assets sub-directory, or create one if it doesn't exist."
    assets_destination_dir = get_assets_dir(export_dir)
    file_io.recursive_create_dir(assets_destination_dir)
    return assets_destination_dir

def get_assets_dir(export_dir):
    if False:
        i = 10
        return i + 15
    'Return path to asset directory in the SavedModel.'
    return file_io.join(compat.as_text(export_dir), compat.as_text(constants.ASSETS_DIRECTORY))

def get_or_create_debug_dir(export_dir):
    if False:
        for i in range(10):
            print('nop')
    'Returns path to the debug sub-directory, creating if it does not exist.'
    debug_dir = get_debug_dir(export_dir)
    file_io.recursive_create_dir(debug_dir)
    return debug_dir

def get_saved_model_pbtxt_path(export_dir):
    if False:
        return 10
    return file_io.join(compat.as_bytes(compat.path_to_str(export_dir)), compat.as_bytes(constants.SAVED_MODEL_FILENAME_PBTXT))

def get_saved_model_pb_path(export_dir):
    if False:
        i = 10
        return i + 15
    return file_io.join(compat.as_bytes(compat.path_to_str(export_dir)), compat.as_bytes(constants.SAVED_MODEL_FILENAME_PB))

def get_debug_dir(export_dir):
    if False:
        for i in range(10):
            print('nop')
    'Returns path to the debug sub-directory in the SavedModel.'
    return file_io.join(compat.as_text(export_dir), compat.as_text(constants.DEBUG_DIRECTORY))