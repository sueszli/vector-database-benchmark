"""Service functions to set and retrieve data from the memory cache."""
from __future__ import annotations
import json
from core.domain import caching_domain
from core.domain import collection_domain
from core.domain import config_domain
from core.domain import exp_domain
from core.domain import platform_parameter_domain
from core.domain import skill_domain
from core.domain import story_domain
from core.domain import topic_domain
from core.platform import models
from typing import Callable, Dict, Final, List, Literal, Mapping, Optional, TypedDict, Union, overload
MYPY = False
if MYPY:
    from mypy_imports import memory_cache_services
    AllowedDefaultTypes = Union[str, int, List[Optional[bool]], Dict[str, float]]
    AllowedCacheableObjectTypes = Union[AllowedDefaultTypes, config_domain.AllowedDefaultValueTypes, collection_domain.Collection, exp_domain.Exploration, skill_domain.Skill, story_domain.Story, topic_domain.Topic, platform_parameter_domain.PlatformParameter]
memory_cache_services = models.Registry.import_cache_services()
MEMCACHE_KEY_DELIMITER = ':'
CACHE_NAMESPACE_EXPLORATION: Final = 'exploration'
CACHE_NAMESPACE_COLLECTION: Final = 'collection'
CACHE_NAMESPACE_SKILL: Final = 'skill'
CACHE_NAMESPACE_STORY: Final = 'story'
CACHE_NAMESPACE_TOPIC: Final = 'topic'
CACHE_NAMESPACE_PLATFORM_PARAMETER: Final = 'platform'
CACHE_NAMESPACE_CONFIG: Final = 'config'
CACHE_NAMESPACE_DEFAULT: Final = 'default'

class DeserializationFunctionsDict(TypedDict):
    """Type for the DESERIALIZATION_FUNCTIONS."""
    collection: Callable[[str], collection_domain.Collection]
    exploration: Callable[[str], exp_domain.Exploration]
    skill: Callable[[str], skill_domain.Skill]
    story: Callable[[str], story_domain.Story]
    topic: Callable[[str], topic_domain.Topic]
    platform: Callable[[str], platform_parameter_domain.PlatformParameter]
    config: Callable[[str], config_domain.AllowedDefaultValueTypes]
    default: Callable[[str], str]

class SerializationFunctionsDict(TypedDict):
    """Type for the SERIALIZATION_FUNCTIONS."""
    collection: Callable[[collection_domain.Collection], str]
    exploration: Callable[[exp_domain.Exploration], str]
    skill: Callable[[skill_domain.Skill], str]
    story: Callable[[story_domain.Story], str]
    topic: Callable[[topic_domain.Topic], str]
    platform: Callable[[platform_parameter_domain.PlatformParameter], str]
    config: Callable[[config_domain.AllowedDefaultValueTypes], str]
    default: Callable[[str], str]
NamespaceType = Literal['collection', 'exploration', 'skill', 'story', 'topic', 'platform', 'config', 'default']
DESERIALIZATION_FUNCTIONS: DeserializationFunctionsDict = {CACHE_NAMESPACE_COLLECTION: collection_domain.Collection.deserialize, CACHE_NAMESPACE_EXPLORATION: exp_domain.Exploration.deserialize, CACHE_NAMESPACE_SKILL: skill_domain.Skill.deserialize, CACHE_NAMESPACE_STORY: story_domain.Story.deserialize, CACHE_NAMESPACE_TOPIC: topic_domain.Topic.deserialize, CACHE_NAMESPACE_PLATFORM_PARAMETER: platform_parameter_domain.PlatformParameter.deserialize, CACHE_NAMESPACE_CONFIG: json.loads, CACHE_NAMESPACE_DEFAULT: json.loads}
SERIALIZATION_FUNCTIONS: SerializationFunctionsDict = {CACHE_NAMESPACE_COLLECTION: lambda x: x.serialize(), CACHE_NAMESPACE_EXPLORATION: lambda x: x.serialize(), CACHE_NAMESPACE_SKILL: lambda x: x.serialize(), CACHE_NAMESPACE_STORY: lambda x: x.serialize(), CACHE_NAMESPACE_TOPIC: lambda x: x.serialize(), CACHE_NAMESPACE_PLATFORM_PARAMETER: lambda x: x.serialize(), CACHE_NAMESPACE_CONFIG: json.dumps, CACHE_NAMESPACE_DEFAULT: json.dumps}

def _get_memcache_key(namespace: NamespaceType, sub_namespace: str | None, obj_id: str) -> str:
    if False:
        while True:
            i = 10
    "Returns a memcache key for the class under the corresponding\n    namespace and sub_namespace.\n\n    Args:\n        namespace: str. The namespace under which the values associated with the\n            id lie. Use CACHE_NAMESPACE_DEFAULT as the namespace for ids that\n            are not associated with a conceptual domain-layer entity and\n            therefore don't require serialization.\n        sub_namespace: str|None. The sub-namespace further differentiates the\n            values. For Explorations, Skills, Stories, Topics, and Collections,\n            the sub-namespace is the stringified version number of the objects.\n        obj_id: str. The id of the value to store in the memory cache.\n\n    Raises:\n        ValueError. The sub-namespace contains a ':'.\n\n    Returns:\n        str. The generated key for use in the memory cache in order to\n        differentiate a passed-in key based on namespace and sub-namespace.\n    "
    sub_namespace_key_string = sub_namespace or ''
    if MEMCACHE_KEY_DELIMITER in sub_namespace_key_string:
        raise ValueError("Sub-namespace %s cannot contain ':'." % sub_namespace_key_string)
    return '%s%s%s%s%s' % (namespace, MEMCACHE_KEY_DELIMITER, sub_namespace_key_string, MEMCACHE_KEY_DELIMITER, obj_id)

def flush_memory_caches() -> None:
    if False:
        while True:
            i = 10
    'Flushes the memory caches by wiping all of the data.'
    memory_cache_services.flush_caches()

@overload
def get_multi(namespace: Literal['collection'], sub_namespace: str | None, obj_ids: List[str]) -> Dict[str, collection_domain.Collection]:
    if False:
        return 10
    ...

@overload
def get_multi(namespace: Literal['exploration'], sub_namespace: str | None, obj_ids: List[str]) -> Dict[str, exp_domain.Exploration]:
    if False:
        while True:
            i = 10
    ...

@overload
def get_multi(namespace: Literal['skill'], sub_namespace: str | None, obj_ids: List[str]) -> Dict[str, skill_domain.Skill]:
    if False:
        return 10
    ...

@overload
def get_multi(namespace: Literal['story'], sub_namespace: str | None, obj_ids: List[str]) -> Dict[str, story_domain.Story]:
    if False:
        while True:
            i = 10
    ...

@overload
def get_multi(namespace: Literal['topic'], sub_namespace: str | None, obj_ids: List[str]) -> Dict[str, topic_domain.Topic]:
    if False:
        while True:
            i = 10
    ...

@overload
def get_multi(namespace: Literal['platform'], sub_namespace: str | None, obj_ids: List[str]) -> Dict[str, platform_parameter_domain.PlatformParameter]:
    if False:
        for i in range(10):
            print('nop')
    ...

@overload
def get_multi(namespace: Literal['config'], sub_namespace: str | None, obj_ids: List[str]) -> Dict[str, config_domain.AllowedDefaultValueTypes]:
    if False:
        print('Hello World!')
    ...

@overload
def get_multi(namespace: Literal['default'], sub_namespace: str | None, obj_ids: List[str]) -> Dict[str, AllowedDefaultTypes]:
    if False:
        return 10
    ...

def get_multi(namespace: NamespaceType, sub_namespace: str | None, obj_ids: List[str]) -> Mapping[str, AllowedCacheableObjectTypes]:
    if False:
        print('Hello World!')
    "Get a dictionary of the {id, value} pairs from the memory cache.\n\n    Args:\n        namespace: str. The namespace under which the values associated with\n            these object ids lie. The namespace determines how the objects are\n            decoded from their JSON-encoded string. Use CACHE_NAMESPACE_DEFAULT\n            as the namespace for objects that are not associated with a\n            conceptual domain-layer entity and therefore don't require\n            serialization.\n        sub_namespace: str|None. The sub-namespace further differentiates the\n            values. For Explorations, Skills, Stories, Topics, and Collections,\n            the sub-namespace is either None or the stringified version number\n            of the objects. If the sub-namespace is not required, pass in None.\n        obj_ids: list(str). List of object ids corresponding to values to\n            retrieve from the cache.\n\n    Raises:\n        ValueError. The namespace does not exist or is not recognized.\n\n    Returns:\n        dict(str, Exploration|Skill|Story|Topic|Collection|str). Dictionary of\n        decoded (id, value) pairs retrieved from the platform caching service.\n    "
    result_dict: Dict[str, AllowedCacheableObjectTypes] = {}
    if len(obj_ids) == 0:
        return result_dict
    if namespace not in DESERIALIZATION_FUNCTIONS:
        raise ValueError('Invalid namespace: %s.' % namespace)
    memcache_keys = [_get_memcache_key(namespace, sub_namespace, obj_id) for obj_id in obj_ids]
    values = memory_cache_services.get_multi(memcache_keys)
    for (obj_id, value) in zip(obj_ids, values):
        if value:
            result_dict[obj_id] = DESERIALIZATION_FUNCTIONS[namespace](value)
    return result_dict

@overload
def set_multi(namespace: Literal['exploration'], sub_namespace: str | None, id_value_mapping: Dict[str, exp_domain.Exploration]) -> bool:
    if False:
        for i in range(10):
            print('nop')
    ...

@overload
def set_multi(namespace: Literal['collection'], sub_namespace: str | None, id_value_mapping: Dict[str, collection_domain.Collection]) -> bool:
    if False:
        print('Hello World!')
    ...

@overload
def set_multi(namespace: Literal['skill'], sub_namespace: str | None, id_value_mapping: Dict[str, skill_domain.Skill]) -> bool:
    if False:
        i = 10
        return i + 15
    ...

@overload
def set_multi(namespace: Literal['story'], sub_namespace: str | None, id_value_mapping: Dict[str, story_domain.Story]) -> bool:
    if False:
        for i in range(10):
            print('nop')
    ...

@overload
def set_multi(namespace: Literal['topic'], sub_namespace: str | None, id_value_mapping: Dict[str, topic_domain.Topic]) -> bool:
    if False:
        print('Hello World!')
    ...

@overload
def set_multi(namespace: Literal['platform'], sub_namespace: str | None, id_value_mapping: Dict[str, platform_parameter_domain.PlatformParameter]) -> bool:
    if False:
        print('Hello World!')
    ...

@overload
def set_multi(namespace: Literal['config'], sub_namespace: str | None, id_value_mapping: Dict[str, config_domain.AllowedDefaultValueTypes]) -> bool:
    if False:
        print('Hello World!')
    ...

@overload
def set_multi(namespace: Literal['default'], sub_namespace: str | None, id_value_mapping: Mapping[str, AllowedDefaultTypes]) -> bool:
    if False:
        print('Hello World!')
    ...

def set_multi(namespace: NamespaceType, sub_namespace: str | None, id_value_mapping: Mapping[str, AllowedCacheableObjectTypes]) -> bool:
    if False:
        i = 10
        return i + 15
    "Set multiple id values at once to the cache, where the values are all\n    of a specific namespace type or a Redis compatible type (more details here:\n    https://redis.io/topics/data-types).\n\n    Args:\n        namespace: str. The namespace under which the values associated with the\n            id lie. Use CACHE_NAMESPACE_DEFAULT as the namespace for objects\n            that are not associated with a conceptual domain-layer entity and\n            therefore don't require serialization.\n        sub_namespace: str|None. The sub-namespace further differentiates the\n            values. For Explorations, Skills, Stories, Topics, and Collections,\n            the sub-namespace is either None or the stringified version number\n            of the objects. If the sub-namespace is not required, pass in None.\n        id_value_mapping:\n            dict(str, Exploration|Skill|Story|Topic|Collection|str). A dict of\n            {id, value} pairs to set to the cache.\n\n    Raises:\n        ValueError. The namespace does not exist or is not recognized.\n\n    Returns:\n        bool. Whether all operations complete successfully.\n    "
    if len(id_value_mapping) == 0:
        return True
    memory_cache_id_value_mapping = {_get_memcache_key(namespace, sub_namespace, obj_id): SERIALIZATION_FUNCTIONS[namespace](value) for (obj_id, value) in id_value_mapping.items()}
    return memory_cache_services.set_multi(memory_cache_id_value_mapping)

def delete_multi(namespace: NamespaceType, sub_namespace: str | None, obj_ids: List[str]) -> bool:
    if False:
        return 10
    "Deletes multiple ids in the cache.\n\n    Args:\n        namespace: str. The namespace under which the values associated with the\n            id lie. Use CACHE_NAMESPACE_DEFAULT namespace for object ids that\n            are not associated with a conceptual domain-layer entity and\n            therefore don't require serialization.\n        sub_namespace: str|None. The sub-namespace further differentiates the\n            values. For Explorations, Skills, Stories, Topics, and Collections,\n            the sub-namespace is either None or the stringified version number\n            of the objects. If the sub-namespace is not required, pass in None.\n        obj_ids: list(str). A list of id strings to delete from the cache.\n\n    Raises:\n        ValueError. The namespace does not exist or is not recognized.\n\n    Returns:\n        bool. Whether all operations complete successfully.\n    "
    if len(obj_ids) == 0:
        return True
    memcache_keys = [_get_memcache_key(namespace, sub_namespace, obj_id) for obj_id in obj_ids]
    return memory_cache_services.delete_multi(memcache_keys) == len(obj_ids)

def get_memory_cache_stats() -> caching_domain.MemoryCacheStats:
    if False:
        i = 10
        return i + 15
    'Get a memory profile of the cache in a dictionary dependent on how the\n    caching service profiles its own cache.\n\n    Returns:\n        MemoryCacheStats. MemoryCacheStats object containing the total allocated\n        memory in bytes, peak memory usage in bytes, and the total number of\n        keys stored as values.\n    '
    return memory_cache_services.get_memory_cache_stats()