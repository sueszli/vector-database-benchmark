from __future__ import annotations
from collections.abc import Iterable
from ansible.module_utils.six import string_types
from ansible.utils.display import Display
display = Display()
__all__ = ['listify_lookup_plugin_terms']

def listify_lookup_plugin_terms(terms, templar, loader=None, fail_on_undefined=True, convert_bare=False):
    if False:
        print('Hello World!')
    if loader is not None:
        display.deprecated('"listify_lookup_plugin_terms" does not use "dataloader" anymore, the ability to pass it in will be removed in future versions.', version='2.18')
    if isinstance(terms, string_types):
        terms = templar.template(terms.strip(), convert_bare=convert_bare, fail_on_undefined=fail_on_undefined)
    else:
        terms = templar.template(terms, fail_on_undefined=fail_on_undefined)
    if isinstance(terms, string_types) or not isinstance(terms, Iterable):
        terms = [terms]
    return terms