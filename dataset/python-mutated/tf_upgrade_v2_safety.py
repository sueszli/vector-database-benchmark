"""Upgrader for Python scripts from 1.* to 2.0 TensorFlow using SAFETY mode."""
from tensorflow.tools.compatibility import all_renames_v2
from tensorflow.tools.compatibility import ast_edits
from tensorflow.tools.compatibility import module_deprecations_v2

class TFAPIChangeSpec(ast_edits.APIChangeSpec):
    """List of maps that describe what changed in the API."""

    def __init__(self):
        if False:
            return 10
        self.function_keyword_renames = {}
        self.symbol_renames = {}
        self.change_to_function = {}
        self.function_reorders = {}
        self.function_warnings = {}
        self.function_transformers = {}
        self.module_deprecations = module_deprecations_v2.MODULE_DEPRECATIONS
        for (symbol, replacement) in all_renames_v2.addons_symbol_mappings.items():
            warning = (ast_edits.WARNING, '(Manual edit required) `{}` has been migrated to `{}` in TensorFlow Addons. The API spec may have changed during the migration. Please see https://github.com/tensorflow/addons for more info.'.format(symbol, replacement))
            self.function_warnings[symbol] = warning
        self.import_renames = {'tensorflow': ast_edits.ImportRename('tensorflow.compat.v1', excluded_prefixes=['tensorflow.contrib', 'tensorflow.flags', 'tensorflow.compat', 'tensorflow.compat.v1', 'tensorflow.compat.v2', 'tensorflow.google'])}
        self.max_submodule_depth = 2