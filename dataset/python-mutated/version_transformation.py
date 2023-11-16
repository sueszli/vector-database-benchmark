import copy
import logging
from collections import defaultdict
from functools import total_ordering
from typing import Callable, Dict, List, Optional
from packaging import version as pkg_version
logger = logging.getLogger(__name__)

@total_ordering
class VersionTransformation:
    """Wrapper class for transformations to config dicts."""

    def __init__(self, transform: Callable[[Dict], Dict], version: str, prefixes: List[str]=None):
        if False:
            while True:
                i = 10
        'Constructor.\n\n        Args:\n            transform: A function or other callable from Dict -> Dict which returns a modified version of the config.\n                       The callable may update the config in-place and return it, or return a new dict.\n            version: The Ludwig version, should be the first version which requires this transform.\n            prefixes: A list of config prefixes this transform should apply to, i.e. ["hyperopt"].  If not specified,\n                      transform will be called with the entire config dictionary.\n        '
        self.transform = transform
        self.version = version
        self.pkg_version = pkg_version.parse(version)
        self.prefixes = prefixes if prefixes else []

    def transform_config(self, config: Dict):
        if False:
            for i in range(10):
                print('nop')
        'Transforms the sepcified config, returns the transformed config.'
        prefixes = self.prefixes if self.prefixes else ['']
        for prefix in prefixes:
            if prefix and (prefix not in config or not config[prefix]):
                continue
            config = self.transform_config_with_prefix(config, prefix)
        return config

    def transform_config_with_prefix(self, config: Dict, prefix: Optional[str]=None) -> Dict:
        if False:
            for i in range(10):
                print('nop')
        'Applied this version transformation to a specified prefix of the config, returns the updated config. If\n        prefix names a list, i.e. "input_features", applies the transformation to each list element (input\n        feature).\n\n        Args:\n            config: A config dictionary.\n            prefix: An optional keypath prefix i.e. "input_features". If no prefix specified, transformation is applied\n                    to config itself.\n\n        Returns The updated config.\n        '
        if prefix:
            components = prefix.split('.', 1)
            key = components[0]
            rest_of_prefix = components[1] if len(components) > 1 else ''
            if key in config:
                subsection = config[key]
                if isinstance(subsection, list):
                    config[key] = [self.transform_config_with_prefix(v, prefix=rest_of_prefix) if isinstance(v, dict) else v for v in subsection]
                elif isinstance(subsection, dict):
                    config[key] = self.transform_config_with_prefix(subsection, prefix=rest_of_prefix)
            return config
        else:
            transformed_config = self.transform(config)
            if transformed_config is None:
                logger.error('Error: version transformation returned None. Check for missing return statement.')
            return transformed_config

    @property
    def max_prefix_length(self):
        if False:
            print('Hello World!')
        'Returns the length of the longest prefix.'
        return max((len(prefix.split('.')) for prefix in self.prefixes)) if self.prefixes else 0

    @property
    def longest_prefix(self):
        if False:
            print('Hello World!')
        'Returns the longest prefix, or empty string if no prefixes specified.'
        prefixes = self.prefixes
        if not prefixes:
            return ''
        max_index = max(range(len(prefixes)), key=lambda i: prefixes[i])
        return prefixes[max_index]

    def __lt__(self, other):
        if False:
            while True:
                i = 10
        'Defines sort order of version transformations. Sorted by:\n\n        - version (ascending)\n        - max_prefix_length (ascending) Process outer config transformations before inner.\n        - longest_prefix (ascending) Order alphabetically by prefix if max_prefix_length equal.\n        '
        return (self.pkg_version, self.max_prefix_length, self.longest_prefix) < (other.pkg_version, other.max_prefix_length, other.longest_prefix)

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        return f'VersionTransformation(<function>, version="{self.version}", prefixes={repr(self.prefixes)})'

class VersionTransformationRegistry:
    """A registry of transformations which update versioned config files."""

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self._registry = defaultdict(list)

    def register(self, transformation: VersionTransformation):
        if False:
            i = 10
            return i + 15
        'Registers a version transformation.'
        self._registry[transformation.version].append(transformation)

    def get_transformations(self, from_version: str, to_version: str) -> List[VersionTransformation]:
        if False:
            while True:
                i = 10
        'Filters transformations to create an ordered list of the config transformations from one version to\n        another. All transformations returned have version st. from_version < version <= to_version.\n\n        Args:\n            from_version: The ludwig version of the input config.\n            to_version: The version to update the config to (usually the current LUDWIG_VERSION).\n\n        Returns an ordered list of transformations to apply to the config to update it.\n        '
        from_version = pkg_version.parse(from_version)
        to_version = pkg_version.parse(to_version)
        to_version = pkg_version.parse(f'{to_version.major}.{to_version.minor}')

        def in_range(v, to_version, from_version):
            if False:
                i = 10
                return i + 15
            v = pkg_version.parse(v)
            return from_version <= v <= to_version
        versions = [v for v in self._registry.keys() if in_range(v, to_version, from_version)]
        transforms = sorted((t for v in versions for t in self._registry[v]))
        return transforms

    def update_config(self, config: Dict, from_version: str, to_version: str) -> Dict:
        if False:
            print('Hello World!')
        'Applies the transformations from an older version to a newer version.\n\n        Args:\n            config: The config, created by ludwig at from_version.\n            from_version: The version of ludwig which wrote the older config.\n            to_version: The version of ludwig to update to (usually the current LUDWIG_VERSION).\n\n        Returns The updated config after applying update transformations and updating the "ludwig_version" key.\n        '
        transformations = self.get_transformations(from_version, to_version)
        updated_config = copy.deepcopy(config)
        for t in transformations:
            updated_config = t.transform_config(updated_config)
        updated_config['ludwig_version'] = to_version
        return updated_config