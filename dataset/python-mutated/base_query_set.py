import abc
from typing import Any
from django.db import router
from django.db.models import QuerySet

class BaseQuerySet(QuerySet, abc.ABC):

    def using_replica(self) -> 'BaseQuerySet':
        if False:
            while True:
                i = 10
        '\n        Use read replica for this query. Database router is expected to use the\n        `replica=True` hint to make routing decision.\n        '
        return self.using(router.db_for_read(self.model, replica=True))

    def defer(self, *args: Any, **kwargs: Any) -> 'BaseQuerySet':
        if False:
            i = 10
            return i + 15
        raise NotImplementedError('Use ``values_list`` instead [performance].')

    def only(self, *args: Any, **kwargs: Any) -> 'BaseQuerySet':
        if False:
            return 10
        raise NotImplementedError('Use ``values_list`` instead [performance].')