from dataclasses import dataclass
from typing import List, Optional, cast
from rest_framework.exceptions import ValidationError
from posthog.constants import PropertyOperatorType
from posthog.models.property import Property, PropertyGroup

@dataclass(frozen=True)
class PropertyGroups:
    outer: Optional[PropertyGroup]
    inner: Optional[PropertyGroup]

class PropertyOptimizer:
    """
    This class is responsible for figuring out what person or group properties can and should be pushed down to their
    respective tables in the query filter.

    This speeds up queries since clickhouse ends up selecting less data.
    """

    def parse_property_groups(self, property_group: PropertyGroup) -> PropertyGroups:
        if False:
            i = 10
            return i + 15
        'Returns outer and inner property groups for persons'
        if len(property_group.values) == 0:
            return PropertyGroups(None, None)
        if property_group.type == PropertyOperatorType.OR:
            if self.using_only_person_properties(property_group):
                return PropertyGroups(None, property_group)
            else:
                return PropertyGroups(property_group, None)
        elif self.using_only_person_properties(property_group):
            return PropertyGroups(None, property_group)
        elif isinstance(property_group.values[0], PropertyGroup):
            outer_property_group_values = []
            inner_property_group_values = []
            for group in property_group.values:
                assert isinstance(group, PropertyGroup)
                subquery_groups = self.parse_property_groups(group)
                if subquery_groups.outer:
                    outer_property_group_values.append(subquery_groups.outer)
                if subquery_groups.inner:
                    inner_property_group_values.append(subquery_groups.inner)
            return PropertyGroups(PropertyGroup(PropertyOperatorType.AND, outer_property_group_values), PropertyGroup(PropertyOperatorType.AND, inner_property_group_values))
        elif isinstance(property_group.values[0], Property):
            outer_property_values = []
            inner_property_values = []
            for property in property_group.values:
                assert isinstance(property, Property)
                if property.type == 'person':
                    inner_property_values.append(property)
                else:
                    outer_property_values.append(property)
            return PropertyGroups(PropertyGroup(PropertyOperatorType.AND, outer_property_values), PropertyGroup(PropertyOperatorType.AND, inner_property_values))
        else:
            raise ValidationError('Invalid property group values')

    @staticmethod
    def using_only_person_properties(property_group: PropertyGroup) -> bool:
        if False:
            while True:
                i = 10
        if len(property_group.values) == 0:
            return True
        if isinstance(property_group.values[0], Property):
            return all((property.type == 'person' for property in property_group.values))
        elif isinstance(property_group.values[0], PropertyGroup):
            return all((PropertyOptimizer.using_only_person_properties(group) for group in cast(List[PropertyGroup], property_group.values)))
        else:
            raise ValidationError('Invalid property group values')