"""Base Factory Abstract Class for Creating Objects Specific to a Resource Type"""
import logging
from abc import ABC, abstractmethod
from typing import Callable, Dict, Generic, List, Optional, TypeVar
from samcli.lib.providers.provider import ResourceIdentifier, Stack, get_resource_by_id
LOG = logging.getLogger(__name__)
T = TypeVar('T')

class ResourceTypeBasedFactory(ABC, Generic[T]):

    def __init__(self, stacks: List[Stack]) -> None:
        if False:
            for i in range(10):
                print('nop')
        self._stacks = stacks

    @abstractmethod
    def _get_generator_mapping(self) -> Dict[str, Callable]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns\n        -------\n        Dict[str, GeneratorFunction]\n            Mapping between resource type and generator function\n        '
        raise NotImplementedError()

    def _get_resource_type(self, resource_identifier: ResourceIdentifier) -> Optional[str]:
        if False:
            print('Hello World!')
        'Get resource type of the resource\n\n        Parameters\n        ----------\n        resource_identifier : ResourceIdentifier\n\n        Returns\n        -------\n        Optional[str]\n            Resource type of the resource\n        '
        resource = get_resource_by_id(self._stacks, resource_identifier)
        if not resource:
            LOG.debug('Resource %s does not exist.', str(resource_identifier))
            return None
        resource_type = resource.get('Type', None)
        if not isinstance(resource_type, str):
            LOG.debug('Resource %s has none string property Type.', str(resource_identifier))
            return None
        return resource_type

    def _get_generator_function(self, resource_identifier: ResourceIdentifier) -> Optional[Callable]:
        if False:
            for i in range(10):
                print('nop')
        'Create an appropriate T object based on stack resource type\n\n        Parameters\n        ----------\n        resource_identifier : ResourceIdentifier\n            Resource identifier of the resource\n\n        Returns\n        -------\n        Optional[T]\n            Object T for the resource. Returns None if resource cannot be\n            found or have no associating T generator function.\n        '
        resource_type = self._get_resource_type(resource_identifier)
        if not resource_type:
            LOG.debug('Resource %s has invalid property Type.', str(resource_identifier))
            return None
        generator = self._get_generator_mapping().get(resource_type, None)
        return generator