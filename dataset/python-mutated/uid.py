"""
Module containing model UID related utility functions.
"""
from __future__ import absolute_import
from st2common.models.db.stormbase import UIDFieldMixin
__all__ = ['parse_uid']

def parse_uid(uid):
    if False:
        while True:
            i = 10
    '\n    Parse UID string.\n\n    :return: (ResourceType, uid_remainder)\n    :rtype: ``tuple``\n    '
    if UIDFieldMixin.UID_SEPARATOR not in uid:
        raise ValueError('Invalid uid: %s' % uid)
    parsed = uid.split(UIDFieldMixin.UID_SEPARATOR)
    if len(parsed) < 2:
        raise ValueError('Invalid or malformed uid: %s' % uid)
    resource_type = parsed[0]
    uid_remainder = parsed[1:]
    return (resource_type, uid_remainder)