"""Contains utility functions used by summary ops."""
import contextlib
import re
from tensorflow.python.framework import ops
from tensorflow.python.platform import tf_logging

def collect(val, collections, default_collections):
    if False:
        for i in range(10):
            print('nop')
    'Adds keys to a collection.\n\n  Args:\n    val: The value to add per each key.\n    collections: A collection of keys to add.\n    default_collections: Used if collections is None.\n  '
    if collections is None:
        collections = default_collections
    for key in collections:
        ops.add_to_collection(key, val)
_INVALID_TAG_CHARACTERS = re.compile('[^-/\\w\\.]')

def clean_tag(name):
    if False:
        while True:
            i = 10
    'Cleans a tag. Removes illegal characters for instance.\n\n  Args:\n    name: The original tag name to be processed.\n\n  Returns:\n    The cleaned tag name.\n  '
    if name is not None:
        new_name = _INVALID_TAG_CHARACTERS.sub('_', name)
        new_name = new_name.lstrip('/')
        if new_name != name:
            tf_logging.info('Summary name %s is illegal; using %s instead.' % (name, new_name))
            name = new_name
    return name

@contextlib.contextmanager
def summary_scope(name, family=None, default_name=None, values=None):
    if False:
        while True:
            i = 10
    "Enters a scope used for the summary and yields both the name and tag.\n\n  To ensure that the summary tag name is always unique, we create a name scope\n  based on `name` and use the full scope name in the tag.\n\n  If `family` is set, then the tag name will be '<family>/<scope_name>', where\n  `scope_name` is `<outer_scope>/<family>/<name>`. This ensures that `family`\n  is always the prefix of the tag (and unmodified), while ensuring the scope\n  respects the outer scope from this summary was created.\n\n  Args:\n    name: A name for the generated summary node.\n    family: Optional; if provided, used as the prefix of the summary tag name.\n    default_name: Optional; if provided, used as default name of the summary.\n    values: Optional; passed as `values` parameter to name_scope.\n\n  Yields:\n    A tuple `(tag, scope)`, both of which are unique and should be used for the\n    tag and the scope for the summary to output.\n  "
    name = clean_tag(name)
    family = clean_tag(family)
    scope_base_name = name if family is None else '{}/{}'.format(family, name)
    with ops.name_scope(scope_base_name, default_name, values, skip_on_eager=False) as scope:
        if family is None:
            tag = scope.rstrip('/')
        else:
            tag = '{}/{}'.format(family, scope.rstrip('/'))
        yield (tag, scope)