from __future__ import absolute_import
from st2common import transport
from st2common.models.db.liveaction import liveaction_access
from st2common.persistence import base as persistence
__all__ = ['LiveAction']

class LiveAction(persistence.StatusBasedResource):
    impl = liveaction_access
    publisher = None

    @classmethod
    def _get_impl(cls):
        if False:
            for i in range(10):
                print('nop')
        return cls.impl

    @classmethod
    def _get_publisher(cls):
        if False:
            while True:
                i = 10
        if not cls.publisher:
            cls.publisher = transport.liveaction.LiveActionPublisher()
        return cls.publisher

    @classmethod
    def delete_by_query(cls, *args, **query):
        if False:
            return 10
        return cls._get_impl().delete_by_query(*args, **query)