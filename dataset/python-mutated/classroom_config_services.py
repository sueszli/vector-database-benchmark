"""Commands for operations on classrooms."""
from __future__ import annotations
from core.constants import constants
from core.domain import classroom_config_domain
from core.platform import models
from typing import Dict, List, Literal, Optional, overload
MYPY = False
if MYPY:
    from mypy_imports import classroom_models
(classroom_models,) = models.Registry.import_models([models.Names.CLASSROOM])

def get_all_classrooms() -> List[classroom_config_domain.Classroom]:
    if False:
        return 10
    'Returns all the classrooms present in the datastore.\n\n    Returns:\n        list(Classroom). The list of classrooms present in the datastore.\n    '
    backend_classroom_models = classroom_models.ClassroomModel.get_all()
    classrooms: List[classroom_config_domain.Classroom] = [get_classroom_from_classroom_model(model) for model in backend_classroom_models]
    return classrooms

def get_classroom_id_to_classroom_name_dict() -> Dict[str, str]:
    if False:
        while True:
            i = 10
    'Returns a dict with classroom id as key and classroom name as value for\n    all the classrooms present in the datastore.\n\n    Returns:\n        dict(str, str). A dict with classroom id as key and classroom name as\n        value for all the classrooms present in the datastore.\n    '
    classrooms = get_all_classrooms()
    return {classroom.classroom_id: classroom.name for classroom in classrooms}

def get_classroom_from_classroom_model(classroom_model: classroom_models.ClassroomModel) -> classroom_config_domain.Classroom:
    if False:
        for i in range(10):
            print('nop')
    'Returns a classroom domain object given a classroom model loaded\n    from the datastore.\n\n    Args:\n        classroom_model: ClassroomModel. The classroom model loaded from the\n            datastore.\n\n    Returns:\n        Classroom. A classroom domain object corresponding to the given\n        classroom model.\n    '
    return classroom_config_domain.Classroom(classroom_model.id, classroom_model.name, classroom_model.url_fragment, classroom_model.course_details, classroom_model.topic_list_intro, classroom_model.topic_id_to_prerequisite_topic_ids)

@overload
def get_classroom_by_id(classroom_id: str) -> classroom_config_domain.Classroom:
    if False:
        i = 10
        return i + 15
    ...

@overload
def get_classroom_by_id(classroom_id: str, *, strict: Literal[True]) -> classroom_config_domain.Classroom:
    if False:
        return 10
    ...

@overload
def get_classroom_by_id(classroom_id: str, *, strict: Literal[False]) -> Optional[classroom_config_domain.Classroom]:
    if False:
        for i in range(10):
            print('nop')
    ...

def get_classroom_by_id(classroom_id: str, strict: bool=True) -> Optional[classroom_config_domain.Classroom]:
    if False:
        while True:
            i = 10
    "Returns a domain object representing a classroom.\n\n    Args:\n        classroom_id: str. ID of the classroom.\n        strict: bool. Fails noisily if the model doesn't exist.\n\n    Returns:\n        Classroom or None. The domain object representing a classroom with the\n        given id, or None if it does not exist.\n    "
    classroom_model = classroom_models.ClassroomModel.get(classroom_id, strict=strict)
    if classroom_model:
        return get_classroom_from_classroom_model(classroom_model)
    else:
        return None

def get_classroom_by_url_fragment(url_fragment: str) -> Optional[classroom_config_domain.Classroom]:
    if False:
        i = 10
        return i + 15
    'Returns a domain object representing a classroom.\n\n    Args:\n        url_fragment: str. The url fragment of the classroom.\n\n    Returns:\n        Classroom or None. The domain object representing a classroom with the\n        given id, or None if it does not exist.\n    '
    classroom_model = classroom_models.ClassroomModel.get_by_url_fragment(url_fragment)
    if classroom_model:
        return get_classroom_from_classroom_model(classroom_model)
    else:
        return None

def get_classroom_url_fragment_for_topic_id(topic_id: str) -> str:
    if False:
        for i in range(10):
            print('nop')
    'Returns the classroom url fragment for the provided topic id.\n\n    Args:\n        topic_id: str. The topic id.\n\n    Returns:\n        str. Returns the classroom url fragment for a topic.\n    '
    classrooms = get_all_classrooms()
    for classroom in classrooms:
        topic_ids = list(classroom.topic_id_to_prerequisite_topic_ids.keys())
        if topic_id in topic_ids:
            return classroom.url_fragment
    return str(constants.CLASSROOM_URL_FRAGMENT_FOR_UNATTACHED_TOPICS)

def get_new_classroom_id() -> str:
    if False:
        while True:
            i = 10
    'Returns a new classroom ID.\n\n    Returns:\n        str. A new classroom ID.\n    '
    return classroom_models.ClassroomModel.generate_new_classroom_id()

def update_classroom(classroom: classroom_config_domain.Classroom, classroom_model: classroom_models.ClassroomModel) -> None:
    if False:
        while True:
            i = 10
    'Saves a Clasroom domain object to the datastore.\n\n    Args:\n        classroom: Classroom. The classroom domain object for the given\n            classroom.\n        classroom_model: ClassroomModel. The classroom model instance.\n    '
    classroom.validate()
    classroom_model.name = classroom.name
    classroom_model.url_fragment = classroom.url_fragment
    classroom_model.course_details = classroom.course_details
    classroom_model.topic_list_intro = classroom.topic_list_intro
    classroom_model.topic_id_to_prerequisite_topic_ids = classroom.topic_id_to_prerequisite_topic_ids
    classroom_model.update_timestamps()
    classroom_model.put()

def create_new_classroom(classroom: classroom_config_domain.Classroom) -> None:
    if False:
        print('Hello World!')
    'Creates a new classroom model from using the classroom domain object.\n\n    Args:\n        classroom: Classroom. The classroom domain object for the given\n            classroom.\n    '
    classroom.validate()
    classroom_models.ClassroomModel.create(classroom.classroom_id, classroom.name, classroom.url_fragment, classroom.course_details, classroom.topic_list_intro, classroom.topic_id_to_prerequisite_topic_ids)

def update_or_create_classroom_model(classroom: classroom_config_domain.Classroom) -> None:
    if False:
        while True:
            i = 10
    'Updates the properties of an existing classroom model or creates a new\n    classroom model.\n    '
    model = classroom_models.ClassroomModel.get(classroom.classroom_id, strict=False)
    if model is None:
        create_new_classroom(classroom)
    else:
        update_classroom(classroom, model)

def delete_classroom(classroom_id: str) -> None:
    if False:
        while True:
            i = 10
    'Deletes the classroom model.\n\n    Args:\n        classroom_id: str. ID of the classroom which is to be deleted.\n    '
    classroom_models.ClassroomModel.get(classroom_id).delete()