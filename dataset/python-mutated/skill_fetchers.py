"""Getter commands for for skill models."""
from __future__ import annotations
import copy
from core import feconf
from core.domain import caching_services
from core.domain import skill_domain
from core.platform import models
from typing import List, Literal, Optional, overload
MYPY = False
if MYPY:
    from mypy_imports import skill_models
(skill_models,) = models.Registry.import_models([models.Names.SKILL])

def get_multi_skills(skill_ids: List[str], strict: bool=True) -> List[skill_domain.Skill]:
    if False:
        return 10
    "Returns a list of skills matching the skill IDs provided.\n\n    Args:\n        skill_ids: list(str). List of skill IDs to get skills for.\n        strict: bool. Whether to raise an error if a skill doesn't exist.\n\n    Returns:\n        list(Skill). The list of skills matching the provided IDs.\n\n    Raises:\n        Exception. No skill exists for given ID.\n    "
    local_skill_models = skill_models.SkillModel.get_multi(skill_ids)
    for (skill_id, skill_model) in zip(skill_ids, local_skill_models):
        if strict and skill_model is None:
            raise Exception('No skill exists for ID %s' % skill_id)
    skills = [get_skill_from_model(skill_model) for skill_model in local_skill_models if skill_model is not None]
    return skills

@overload
def get_skill_by_id(skill_id: str) -> skill_domain.Skill:
    if False:
        print('Hello World!')
    ...

@overload
def get_skill_by_id(skill_id: str, *, version: Optional[int]=None) -> skill_domain.Skill:
    if False:
        for i in range(10):
            print('nop')
    ...

@overload
def get_skill_by_id(skill_id: str, *, strict: Literal[True], version: Optional[int]=None) -> skill_domain.Skill:
    if False:
        i = 10
        return i + 15
    ...

@overload
def get_skill_by_id(skill_id: str, *, strict: Literal[False], version: Optional[int]=None) -> Optional[skill_domain.Skill]:
    if False:
        i = 10
        return i + 15
    ...

def get_skill_by_id(skill_id: str, strict: bool=True, version: Optional[int]=None) -> Optional[skill_domain.Skill]:
    if False:
        while True:
            i = 10
    'Returns a domain object representing a skill.\n\n    Args:\n        skill_id: str. ID of the skill.\n        strict: bool. Whether to fail noisily if no skill with the given\n            id exists in the datastore.\n        version: int or None. The version number of the skill to be\n            retrieved. If it is None, the latest version will be retrieved.\n\n    Returns:\n        Skill or None. The domain object representing a skill with the\n        given id, or None if it does not exist.\n    '
    sub_namespace = str(version) if version else None
    cached_skill = caching_services.get_multi(caching_services.CACHE_NAMESPACE_SKILL, sub_namespace, [skill_id]).get(skill_id)
    if cached_skill is not None:
        return cached_skill
    else:
        skill_model = skill_models.SkillModel.get(skill_id, strict=strict, version=version)
        if skill_model:
            skill = get_skill_from_model(skill_model)
            caching_services.set_multi(caching_services.CACHE_NAMESPACE_SKILL, sub_namespace, {skill_id: skill})
            return skill
        else:
            return None

def get_skill_from_model(skill_model: skill_models.SkillModel) -> skill_domain.Skill:
    if False:
        return 10
    'Returns a skill domain object given a skill model loaded\n    from the datastore.\n\n    Args:\n        skill_model: SkillModel. The skill model loaded from the datastore.\n\n    Returns:\n        skill. A Skill domain object corresponding to the given skill model.\n    '
    versioned_skill_contents: skill_domain.VersionedSkillContentsDict = {'schema_version': skill_model.skill_contents_schema_version, 'skill_contents': copy.deepcopy(skill_model.skill_contents)}
    versioned_misconceptions: skill_domain.VersionedMisconceptionDict = {'schema_version': skill_model.misconceptions_schema_version, 'misconceptions': copy.deepcopy(skill_model.misconceptions)}
    versioned_rubrics: skill_domain.VersionedRubricDict = {'schema_version': skill_model.rubric_schema_version, 'rubrics': copy.deepcopy(skill_model.rubrics)}
    if skill_model.skill_contents_schema_version != feconf.CURRENT_SKILL_CONTENTS_SCHEMA_VERSION:
        _migrate_skill_contents_to_latest_schema(versioned_skill_contents)
    if skill_model.misconceptions_schema_version != feconf.CURRENT_MISCONCEPTIONS_SCHEMA_VERSION:
        _migrate_misconceptions_to_latest_schema(versioned_misconceptions)
    if skill_model.rubric_schema_version != feconf.CURRENT_RUBRIC_SCHEMA_VERSION:
        _migrate_rubrics_to_latest_schema(versioned_rubrics)
    return skill_domain.Skill(skill_model.id, skill_model.description, [skill_domain.Misconception.from_dict(misconception) for misconception in versioned_misconceptions['misconceptions']], [skill_domain.Rubric.from_dict(rubric) for rubric in versioned_rubrics['rubrics']], skill_domain.SkillContents.from_dict(versioned_skill_contents['skill_contents']), versioned_misconceptions['schema_version'], versioned_rubrics['schema_version'], versioned_skill_contents['schema_version'], skill_model.language_code, skill_model.version, skill_model.next_misconception_id, skill_model.superseding_skill_id, skill_model.all_questions_merged, skill_model.prerequisite_skill_ids, skill_model.created_on, skill_model.last_updated)

def get_skill_by_description(description: str) -> Optional[skill_domain.Skill]:
    if False:
        print('Hello World!')
    'Returns a domain object representing a skill.\n\n    Args:\n        description: str. The description of the skill.\n\n    Returns:\n        Skill or None. The domain object representing a skill with the\n        given description, or None if it does not exist.\n    '
    skill_model = skill_models.SkillModel.get_by_description(description)
    return get_skill_from_model(skill_model) if skill_model else None

def _migrate_skill_contents_to_latest_schema(versioned_skill_contents: skill_domain.VersionedSkillContentsDict) -> None:
    if False:
        while True:
            i = 10
    'Holds the responsibility of performing a step-by-step, sequential update\n    of the skill contents structure based on the schema version of the input\n    skill contents dictionary. If the current skill_contents schema changes, a\n    new conversion function must be added and some code appended to this\n    function to account for that new version.\n\n    Args:\n        versioned_skill_contents: dict. A dict with two keys:\n            - schema_version: int. The schema version for the skill_contents\n                dict.\n            - skill_contents: dict. The dict comprising the skill contents.\n\n    Raises:\n        Exception. The schema version of the skill_contents is outside of what\n            is supported at present.\n    '
    skill_contents_schema_version = versioned_skill_contents['schema_version']
    if not 1 <= skill_contents_schema_version <= feconf.CURRENT_SKILL_CONTENTS_SCHEMA_VERSION:
        raise Exception('Sorry, we can only process v1-v%d skill schemas at present.' % feconf.CURRENT_SKILL_CONTENTS_SCHEMA_VERSION)
    while skill_contents_schema_version < feconf.CURRENT_SKILL_CONTENTS_SCHEMA_VERSION:
        skill_domain.Skill.update_skill_contents_from_model(versioned_skill_contents, skill_contents_schema_version)
        skill_contents_schema_version += 1

def _migrate_misconceptions_to_latest_schema(versioned_misconceptions: skill_domain.VersionedMisconceptionDict) -> None:
    if False:
        while True:
            i = 10
    'Holds the responsibility of performing a step-by-step, sequential update\n    of the misconceptions structure based on the schema version of the input\n    misconceptions dictionary. If the current misconceptions schema changes, a\n    new conversion function must be added and some code appended to this\n    function to account for that new version.\n\n    Args:\n        versioned_misconceptions: dict. A dict with two keys:\n            - schema_version: int. The schema version for the misconceptions\n                dict.\n            - misconceptions: list(dict). The list of dicts comprising the skill\n                misconceptions.\n\n    Raises:\n        Exception. The schema version of misconceptions is outside of what\n            is supported at present.\n    '
    misconception_schema_version = versioned_misconceptions['schema_version']
    if not 1 <= misconception_schema_version <= feconf.CURRENT_MISCONCEPTIONS_SCHEMA_VERSION:
        raise Exception('Sorry, we can only process v1-v%d misconception schemas at present.' % feconf.CURRENT_MISCONCEPTIONS_SCHEMA_VERSION)
    while misconception_schema_version < feconf.CURRENT_MISCONCEPTIONS_SCHEMA_VERSION:
        skill_domain.Skill.update_misconceptions_from_model(versioned_misconceptions, misconception_schema_version)
        misconception_schema_version += 1

def _migrate_rubrics_to_latest_schema(versioned_rubrics: skill_domain.VersionedRubricDict) -> None:
    if False:
        for i in range(10):
            print('nop')
    'Holds the responsibility of performing a step-by-step, sequential update\n    of the rubrics structure based on the schema version of the input\n    rubrics dictionary. If the current rubrics schema changes, a\n    new conversion function must be added and some code appended to this\n    function to account for that new version.\n\n    Args:\n        versioned_rubrics: dict. A dict with two keys:\n            - schema_version: int. The schema version for the rubrics dict.\n            - rubrics: list(dict). The list of dicts comprising the skill\n                rubrics.\n\n    Raises:\n        Exception. The schema version of rubrics is outside of what is supported\n            at present.\n    '
    rubric_schema_version = versioned_rubrics['schema_version']
    if not 1 <= rubric_schema_version <= feconf.CURRENT_RUBRIC_SCHEMA_VERSION:
        raise Exception('Sorry, we can only process v1-v%d rubric schemas at present.' % feconf.CURRENT_RUBRIC_SCHEMA_VERSION)
    while rubric_schema_version < feconf.CURRENT_RUBRIC_SCHEMA_VERSION:
        skill_domain.Skill.update_rubrics_from_model(versioned_rubrics, rubric_schema_version)
        rubric_schema_version += 1