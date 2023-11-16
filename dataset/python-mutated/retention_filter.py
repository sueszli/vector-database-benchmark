import json
from typing import Any, Dict, Optional, Tuple, Union
from rest_framework.request import Request
from posthog.constants import INSIGHT_RETENTION
from .base_filter import BaseFilter
from .mixins.common import BreakdownMixin, ClientQueryIdMixin, DisplayDerivedMixin, FilterTestAccountsMixin, InsightMixin, LimitMixin, OffsetMixin, SampleMixin
from .mixins.funnel import FunnelCorrelationMixin
from .mixins.groups import GroupsAggregationMixin
from .mixins.property import PropertyMixin
from .mixins.retention import EntitiesDerivedMixin, RetentionDateDerivedMixin, RetentionTypeMixin
from .mixins.simplify import SimplifyFilterMixin
from .mixins.utils import cached_property, include_dict
RETENTION_DEFAULT_INTERVALS = 11

class RetentionFilter(RetentionTypeMixin, EntitiesDerivedMixin, RetentionDateDerivedMixin, PropertyMixin, DisplayDerivedMixin, FilterTestAccountsMixin, BreakdownMixin, InsightMixin, OffsetMixin, LimitMixin, GroupsAggregationMixin, FunnelCorrelationMixin, ClientQueryIdMixin, SimplifyFilterMixin, SampleMixin, BaseFilter):

    def __init__(self, data: Dict[str, Any]={}, request: Optional[Request]=None, **kwargs) -> None:
        if False:
            print('Hello World!')
        if data:
            data['insight'] = INSIGHT_RETENTION
        else:
            data = {'insight': INSIGHT_RETENTION}
        super().__init__(data, request, **kwargs)

    @cached_property
    def breakdown_values(self) -> Optional[Tuple[Union[str, int], ...]]:
        if False:
            while True:
                i = 10
        raw_value = self._data.get('breakdown_values', None)
        if raw_value is None:
            return None
        if isinstance(raw_value, str):
            return tuple(json.loads(raw_value))
        return tuple(raw_value)

    @include_dict
    def breakdown_values_to_dict(self):
        if False:
            i = 10
            return i + 15
        return {'breakdown_values': self.breakdown_values} if self.breakdown_values else {}