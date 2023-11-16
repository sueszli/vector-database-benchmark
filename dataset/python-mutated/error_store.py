"""Utilities for storing collections of error messages.

.. warning::

    This module is treated as private API.
    Users should not need to use this module directly.
"""
from marshmallow.exceptions import SCHEMA

class ErrorStore:

    def __init__(self):
        if False:
            return 10
        self.errors = {}

    def store_error(self, messages, field_name=SCHEMA, index=None):
        if False:
            i = 10
            return i + 15
        if field_name != SCHEMA or not isinstance(messages, dict):
            messages = {field_name: messages}
        if index is not None:
            messages = {index: messages}
        self.errors = merge_errors(self.errors, messages)

def merge_errors(errors1, errors2):
    if False:
        i = 10
        return i + 15
    'Deeply merge two error messages.\n\n    The format of ``errors1`` and ``errors2`` matches the ``message``\n    parameter of :exc:`marshmallow.exceptions.ValidationError`.\n    '
    if not errors1:
        return errors2
    if not errors2:
        return errors1
    if isinstance(errors1, list):
        if isinstance(errors2, list):
            return errors1 + errors2
        if isinstance(errors2, dict):
            return dict(errors2, **{SCHEMA: merge_errors(errors1, errors2.get(SCHEMA))})
        return errors1 + [errors2]
    if isinstance(errors1, dict):
        if isinstance(errors2, list):
            return dict(errors1, **{SCHEMA: merge_errors(errors1.get(SCHEMA), errors2)})
        if isinstance(errors2, dict):
            errors = dict(errors1)
            for (key, val) in errors2.items():
                if key in errors:
                    errors[key] = merge_errors(errors[key], val)
                else:
                    errors[key] = val
            return errors
        return dict(errors1, **{SCHEMA: merge_errors(errors1.get(SCHEMA), errors2)})
    if isinstance(errors2, list):
        return [errors1] + errors2
    if isinstance(errors2, dict):
        return dict(errors2, **{SCHEMA: merge_errors(errors1, errors2.get(SCHEMA))})
    return [errors1, errors2]