"""Python strict deprecation mode enabler."""
from tensorflow.python.util.tf_export import tf_export
STRICT_MODE = False

@tf_export('experimental.enable_strict_mode')
def enable_strict_mode():
    if False:
        i = 10
        return i + 15
    'If called, enables strict mode for all behaviors.\n\n  Used to switch all deprecation warnings to raise errors instead.\n  '
    global STRICT_MODE
    STRICT_MODE = True