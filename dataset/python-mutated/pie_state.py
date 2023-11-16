from typing import Any
import reflex as rx
from ..state import State

class PieChartState(State):
    """Pie Chart State."""
    resources: list[dict[str, Any]] = [dict(type_='🏆', count=1), dict(type_='🪵', count=1), dict(type_='🥑', count=1), dict(type_='🧱', count=1)]

    @rx.cached_var
    def resource_types(self) -> list[str]:
        if False:
            while True:
                i = 10
        'Get the resource types.\n\n        Returns:\n            The resource types.\n        '
        return [r['type_'] for r in self.resources]

    def increment(self, type_: str):
        if False:
            return 10
        'Increment the count of a resource type.\n\n        Args:\n            type_: The type of resource to increment.\n        '
        for resource in self.resources:
            if resource['type_'] == type_:
                resource['count'] += 1
                break

    def decrement(self, type_: str):
        if False:
            print('Hello World!')
        'Decrement the count of a resource type.\n\n        Args:\n            type_: The type of resource to decrement.\n        '
        for resource in self.resources:
            if resource['type_'] == type_ and resource['count'] > 0:
                resource['count'] -= 1
                break