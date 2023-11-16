"""Filesystem related operations."""
from tensorflow.python.ops import gen_filesystem_ops as _gen_filesystem_ops

def filesystem_set_configuration(scheme, key, value, name=None):
    if False:
        print('Hello World!')
    'Set configuration of the file system.\n\n  Args:\n    scheme: File system scheme.\n    key: The name of the configuration option.\n    value: The value of the configuration option.\n    name: A name for the operation (optional).\n\n  Returns:\n    None.\n  '
    return _gen_filesystem_ops.file_system_set_configuration(scheme, key=key, value=value, name=name)