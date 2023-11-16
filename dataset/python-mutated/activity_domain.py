"""Domain object for a reference to an activity."""
from __future__ import annotations
from core.constants import constants
from typing import Dict, List

class ActivityReference:
    """Domain object for an activity reference.

    An activity is a piece of learning material that can be created in Oppia.
    Currently, the only available types of activities are explorations and
    collections.

    Attributes:
        type: str. The activity type.
        id: str. The activity id.
    """

    def __init__(self, activity_type: str, activity_id: str) -> None:
        if False:
            return 10
        'Constructs an ActivityReference domain object.\n\n        Args:\n            activity_type: str. The activity type.\n            activity_id: str. The activity id.\n        '
        self.type = activity_type
        self.id = activity_id

    def get_hash(self) -> str:
        if False:
            return 10
        'Returns a unique string for this ActivityReference domain object.\n\n        Returns:\n            str. A unique string hash for this ActivityReference domain object.\n        '
        return '%s:%s' % (self.type, self.id)

    def validate(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Checks that all fields of this ActivityReference domain object\n        are valid.\n\n        Raises:\n            Exception. The activity type is invalid.\n        '
        if self.type not in (constants.ACTIVITY_TYPE_EXPLORATION, constants.ACTIVITY_TYPE_COLLECTION):
            raise Exception('Invalid activity type: %s' % self.type)
        if not isinstance(self.id, str):
            raise Exception('Expected id to be a string but found %s' % self.id)

    def to_dict(self) -> Dict[str, str]:
        if False:
            i = 10
            return i + 15
        "Returns a dict representing this ActivityReference domain object.\n\n        Returns:\n            dict. A dict, mapping type and id of an ActivityReference\n            instance to corresponding keys 'type' and 'id'.\n        "
        return {'type': self.type, 'id': self.id}

    @classmethod
    def from_dict(cls, activity_reference_dict: Dict[str, str]) -> ActivityReference:
        if False:
            print('Hello World!')
        'Return the ActivityReference object from a dict.\n\n        Args:\n            activity_reference_dict: dict. Dictionary representation\n                of the object.\n\n        Returns:\n            ActivityReference. The corresponding ActivityReference object.\n        '
        return cls(activity_reference_dict['type'], activity_reference_dict['id'])

class ActivityReferences:
    """Domain object for a list of activity references.

    Attributes:
        activity_reference_list: list(ActivityReference). A list of
            ActivityReference domain objects.
    """

    def __init__(self, activity_reference_list: List[ActivityReference]):
        if False:
            return 10
        'Constructs an ActivityReferences domain object.\n\n        Args:\n            activity_reference_list: list(ActivityReference). A list of\n                ActivityReference domain objects.\n        '
        self.activity_reference_list = activity_reference_list

    def validate(self) -> None:
        if False:
            return 10
        'Checks that all ActivityReference domain object in\n        self.activity_reference_list are valid.\n\n        Raises:\n            Exception. Any ActivityReference in self.activity_reference_list\n                is invalid.\n        '
        for reference in self.activity_reference_list:
            reference.validate()