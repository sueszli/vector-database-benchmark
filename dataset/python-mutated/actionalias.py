from __future__ import absolute_import
from st2common import transport
from st2common.models.db.actionalias import actionalias_access
from st2common.persistence.base import Access
__all__ = ['ActionAlias']

class ActionAlias(Access):
    impl = actionalias_access

    @classmethod
    def _get_impl(cls):
        if False:
            while True:
                i = 10
        return cls.impl

    @classmethod
    def _get_publisher(cls):
        if False:
            for i in range(10):
                print('nop')
        if not cls.publisher:
            cls.publisher = transport.actionalias.ActionAliasPublisher()
        return cls.publisher