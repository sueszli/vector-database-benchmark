"""Commands for operations on classrooms."""
from __future__ import annotations
from core.constants import constants
from core.domain import classroom_domain
from core.domain import config_domain
from typing import Optional

def get_classroom_url_fragment_for_topic_id(topic_id: str) -> str:
    if False:
        for i in range(10):
            print('nop')
    'Returns the classroom url fragment for the provided topic id.\n\n    Args:\n        topic_id: str. The topic id.\n\n    Returns:\n        str. Returns the classroom url fragment for a topic.\n    '
    for classroom_dict in config_domain.CLASSROOM_PAGES_DATA.value:
        if topic_id in classroom_dict['topic_ids']:
            return str(classroom_dict['url_fragment'])
    return str(constants.CLASSROOM_URL_FRAGMENT_FOR_UNATTACHED_TOPICS)

def get_classroom_by_url_fragment(classroom_url_fragment: str) -> Optional[classroom_domain.Classroom]:
    if False:
        i = 10
        return i + 15
    'Returns the classroom domain object for the provided classroom url\n    fragment.\n\n    Args:\n        classroom_url_fragment: str. The classroom url fragment.\n\n    Returns:\n        Classroom|None. Returns the classroom domain object if found, else\n        returns None.\n    '
    for classroom_dict in config_domain.CLASSROOM_PAGES_DATA.value:
        if classroom_url_fragment == classroom_dict['url_fragment']:
            return classroom_domain.Classroom(classroom_dict['name'], classroom_dict['url_fragment'], classroom_dict['topic_ids'], classroom_dict['course_details'], classroom_dict['topic_list_intro'])
    return None