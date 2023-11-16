"""Domain objects for Classroom."""
from __future__ import annotations
from typing import List, TypedDict

class ClassroomDict(TypedDict):
    """Dictionary representing the Classroom object."""
    name: str
    url_fragment: str
    topic_ids: List[str]
    course_details: str
    topic_list_intro: str

class Classroom:
    """Domain object for a classroom."""

    def __init__(self, name: str, url_fragment: str, topic_ids: List[str], course_details: str, topic_list_intro: str) -> None:
        if False:
            return 10
        'Constructs a Classroom domain object.\n\n        Args:\n            name: str. The name of the classroom.\n            url_fragment: str. The url fragment of the classroom.\n            topic_ids: list(str). List of topic ids attached to the classroom.\n            course_details: str. Course details for the classroom.\n            topic_list_intro: str. Topic list introduction for the classroom.\n        '
        self.name = name
        self.url_fragment = url_fragment
        self.topic_ids = topic_ids
        self.course_details = course_details
        self.topic_list_intro = topic_list_intro

    def to_dict(self) -> ClassroomDict:
        if False:
            while True:
                i = 10
        'Converts this Classroom domain instance into a dictionary form with\n        its keys as the attributes of this class.\n\n        Returns:\n            dict. A dictionary containing the Classroom class information in a\n            dictionary form.\n        '
        return {'name': self.name, 'url_fragment': self.url_fragment, 'topic_ids': self.topic_ids, 'course_details': self.course_details, 'topic_list_intro': self.topic_list_intro}