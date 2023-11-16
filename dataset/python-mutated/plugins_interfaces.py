"""
Provide IAC Plugins Interface && Project representation
"""
import abc
import logging
from collections import OrderedDict
from collections.abc import Mapping, MutableMapping
from copy import deepcopy
from enum import Enum
from typing import Any, Dict, Iterator, List, Optional, Union
from uuid import uuid4
from samcli.lib.utils.packagetype import IMAGE, ZIP
LOG = logging.getLogger(__name__)

class Environment:

    def __init__(self, region: Optional[str]=None, account_id: Optional[str]=None):
        if False:
            print('Hello World!')
        self._region = region
        self._account_id = account_id

    @property
    def region(self) -> Optional[str]:
        if False:
            print('Hello World!')
        return self._region

    @region.setter
    def region(self, region: str) -> None:
        if False:
            for i in range(10):
                print('nop')
        self._region = region

    @property
    def account_id(self) -> Optional[str]:
        if False:
            print('Hello World!')
        return self._account_id

    @account_id.setter
    def account_id(self, account_id: str) -> None:
        if False:
            print('Hello World!')
        self._account_id = account_id

class Destination:

    def __init__(self, path: str, value: Any):
        if False:
            return 10
        self._path = path
        self._value = value

    @property
    def path(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        return self._path

    @path.setter
    def path(self, path: str) -> None:
        if False:
            while True:
                i = 10
        self._path = path

    @property
    def value(self) -> Any:
        if False:
            print('Hello World!')
        return self._value

    @value.setter
    def value(self, value: Any) -> None:
        if False:
            print('Hello World!')
        self._value = value

class Asset:

    def __init__(self, asset_id: Optional[str]=None, destinations: Optional[List[Destination]]=None, source_property: Optional[str]=None, extra_details: Optional[Dict[str, Any]]=None):
        if False:
            i = 10
            return i + 15
        if asset_id is None:
            asset_id = str(uuid4())
        self._asset_id = asset_id
        if destinations is None:
            destinations = []
        self._destinations = destinations
        self._source_property = source_property
        extra_details = extra_details or {}
        self._extra_details = extra_details

    @property
    def asset_id(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        return self._asset_id

    @asset_id.setter
    def asset_id(self, asset_id: str) -> None:
        if False:
            i = 10
            return i + 15
        self._asset_id = asset_id

    @property
    def destinations(self) -> List[Destination]:
        if False:
            return 10
        return self._destinations

    @destinations.setter
    def destinations(self, destinations: List[Destination]) -> None:
        if False:
            i = 10
            return i + 15
        self._destinations = destinations

    @property
    def source_property(self) -> Optional[str]:
        if False:
            i = 10
            return i + 15
        return self._source_property

    @source_property.setter
    def source_property(self, source_property: str) -> None:
        if False:
            for i in range(10):
                print('nop')
        self._source_property = source_property

    @property
    def extra_details(self) -> Dict[str, Any]:
        if False:
            return 10
        return self._extra_details

    @extra_details.setter
    def extra_details(self, extra_details: Dict[str, Any]) -> None:
        if False:
            for i in range(10):
                print('nop')
        self._extra_details = extra_details

class S3Asset(Asset):
    """
    Represent the S3 Assets.
    Can Represent Implicit asset for resources like Lambda Function, or explicit for added assets like HTML folders
    """

    def __init__(self, asset_id: Optional[str]=None, bucket_name: Optional[str]=None, object_key: Optional[str]=None, object_version: Optional[str]=None, source_path: Optional[str]=None, updated_source_path: Optional[str]=None, destinations: Optional[List[Destination]]=None, source_property: Optional[str]=None, extra_details: Optional[Dict[str, Any]]=None):
        if False:
            for i in range(10):
                print('nop')
        self._bucket_name = bucket_name
        self._object_key = object_key
        self._object_version = object_version
        self._source_path = source_path
        self._updated_source_path = updated_source_path
        super().__init__(asset_id, destinations, source_property, extra_details)

    @property
    def bucket_name(self) -> Optional[str]:
        if False:
            print('Hello World!')
        return self._bucket_name

    @bucket_name.setter
    def bucket_name(self, bucket_name: str) -> Optional[None]:
        if False:
            return 10
        self._bucket_name = bucket_name

    @property
    def object_key(self) -> Optional[str]:
        if False:
            i = 10
            return i + 15
        return self._object_key

    @object_key.setter
    def object_key(self, object_key: str) -> None:
        if False:
            print('Hello World!')
        self._object_key = object_key

    @property
    def object_version(self) -> Optional[str]:
        if False:
            print('Hello World!')
        return self._object_version

    @object_version.setter
    def object_version(self, object_version: str) -> None:
        if False:
            while True:
                i = 10
        self._object_version = object_version

    @property
    def source_path(self) -> Optional[str]:
        if False:
            for i in range(10):
                print('nop')
        return self._source_path

    @source_path.setter
    def source_path(self, source_path: str) -> None:
        if False:
            for i in range(10):
                print('nop')
        self._source_path = source_path

    @property
    def updated_source_path(self) -> Optional[str]:
        if False:
            print('Hello World!')
        return self._updated_source_path

    @updated_source_path.setter
    def updated_source_path(self, updated_source_path: str) -> None:
        if False:
            i = 10
            return i + 15
        self._updated_source_path = updated_source_path

class ImageAsset(Asset):
    """
    Represent the Container Assets.
    """

    def __init__(self, asset_id: Optional[str]=None, repository_name: Optional[str]=None, registry: Optional[str]=None, image_tag: Optional[str]=None, source_local_image: Optional[str]=None, source_path: Optional[str]=None, docker_file_name: Optional[str]=None, build_args: Optional[Dict[str, str]]=None, destinations: Optional[List[Destination]]=None, source_property: Optional[str]=None, target: Optional[str]=None, extra_details: Optional[Dict[str, Any]]=None):
        if False:
            i = 10
            return i + 15
        '\n        image uri = <registry>/repository_name:image_tag\n        registry = aws_account_id.dkr.ecr.us-west-2.amazonaws.com\n        '
        self._repository_name = repository_name
        self._registry = registry
        self._image_tag = image_tag
        self._source_local_image = source_local_image
        self._source_path = source_path
        self._docker_file_name = docker_file_name
        self._build_args = build_args
        self._target = target
        super().__init__(asset_id, destinations, source_property, extra_details)

    @property
    def repository_name(self) -> Optional[str]:
        if False:
            while True:
                i = 10
        return self._repository_name

    @repository_name.setter
    def repository_name(self, repository_name: str) -> None:
        if False:
            return 10
        self._repository_name = repository_name

    @property
    def target(self) -> Optional[str]:
        if False:
            i = 10
            return i + 15
        return self._target

    @target.setter
    def target(self, target: str) -> None:
        if False:
            while True:
                i = 10
        self._target = target

    @property
    def build_args(self) -> Optional[Dict[str, str]]:
        if False:
            print('Hello World!')
        return self._build_args

    @build_args.setter
    def build_args(self, build_args: Dict[str, str]) -> None:
        if False:
            return 10
        self._build_args = build_args

    @property
    def registry(self) -> Optional[str]:
        if False:
            return 10
        return self._registry

    @registry.setter
    def registry(self, registry: str) -> None:
        if False:
            i = 10
            return i + 15
        self._registry = registry

    @property
    def image_tag(self) -> Optional[str]:
        if False:
            for i in range(10):
                print('nop')
        return self._image_tag

    @image_tag.setter
    def image_tag(self, image_tag: str) -> None:
        if False:
            while True:
                i = 10
        self._image_tag = image_tag

    @property
    def source_local_image(self) -> Optional[str]:
        if False:
            for i in range(10):
                print('nop')
        return self._source_local_image

    @source_local_image.setter
    def source_local_image(self, source_local_image: str) -> None:
        if False:
            while True:
                i = 10
        self._source_local_image = source_local_image

    @property
    def source_path(self) -> Optional[str]:
        if False:
            while True:
                i = 10
        return self._source_path

    @source_path.setter
    def source_path(self, source_path: str) -> None:
        if False:
            while True:
                i = 10
        self._source_path = source_path

    @property
    def docker_file_name(self) -> Optional[str]:
        if False:
            return 10
        return self._docker_file_name

    @docker_file_name.setter
    def docker_file_name(self, docker_file_name: str) -> None:
        if False:
            return 10
        self._docker_file_name = docker_file_name

class SectionItem:

    def __init__(self, key: Optional[str]=None, item_id: Optional[str]=None):
        if False:
            return 10
        self._key = key
        self._item_id = item_id

    @property
    def key(self) -> Optional[str]:
        if False:
            while True:
                i = 10
        return self._key

    @key.setter
    def key(self, key: str) -> None:
        if False:
            return 10
        self._key = key

    @property
    def item_id(self) -> Optional[str]:
        if False:
            print('Hello World!')
        return self._item_id or self._key

    @item_id.setter
    def item_id(self, item_id: str) -> None:
        if False:
            while True:
                i = 10
        self._item_id = item_id

class SimpleSectionItem(SectionItem):

    def __init__(self, key: Optional[str]=None, item_id: Optional[str]=None, value: Any=None):
        if False:
            i = 10
            return i + 15
        super().__init__(key, item_id)
        self._value = value

    @property
    def value(self) -> Any:
        if False:
            i = 10
            return i + 15
        return self._value

    @value.setter
    def value(self, value: Any) -> None:
        if False:
            print('Hello World!')
        self._value = value

    def __bool__(self) -> bool:
        if False:
            while True:
                i = 10
        return bool(self._value)

class DictSectionItem(SectionItem, MutableMapping):

    def __init__(self, key: Optional[str]=None, item_id: Optional[str]=None, body: Any=None, assets: Optional[List[Asset]]=None, extra_details: Optional[Dict[str, Any]]=None):
        if False:
            while True:
                i = 10
        super().__init__(key, item_id)
        self._body = body or {}
        if assets is None:
            assets = []
        self._assets = assets
        extra_details = extra_details or {}
        self._extra_details = extra_details

    def copy(self) -> 'DictSectionItem':
        if False:
            print('Hello World!')
        return deepcopy(self)

    @property
    def body(self) -> Any:
        if False:
            for i in range(10):
                print('nop')
        return self._body

    @property
    def assets(self) -> List[Asset]:
        if False:
            i = 10
            return i + 15
        return self._assets

    @assets.setter
    def assets(self, assets: List[Asset]) -> None:
        if False:
            print('Hello World!')
        self._assets = assets

    @property
    def extra_details(self) -> Dict[str, Any]:
        if False:
            i = 10
            return i + 15
        return self._extra_details

    @extra_details.setter
    def extra_details(self, extra_details: Dict[str, Any]) -> None:
        if False:
            i = 10
            return i + 15
        self._extra_details = extra_details

    def is_packageable(self) -> bool:
        if False:
            print('Hello World!')
        '\n        return if the resource is packageable\n        '
        return bool(self.assets)

    def find_asset_by_source_property(self, source_property: str) -> Optional[Asset]:
        if False:
            for i in range(10):
                print('nop')
        if not self.assets:
            return None
        for asset in self.assets:
            if asset.source_property == source_property:
                return asset
        return None

    def __setitem__(self, k: str, v: Any) -> None:
        if False:
            print('Hello World!')
        self._body[k] = v

    def __delitem__(self, v: str) -> None:
        if False:
            print('Hello World!')
        del self._body[v]

    def __getitem__(self, k: str) -> Any:
        if False:
            i = 10
            return i + 15
        return self._body[k]

    def __len__(self) -> int:
        if False:
            return 10
        return len(self._body)

    def __iter__(self) -> Iterator:
        if False:
            print('Hello World!')
        return iter(self._body)

    def __bool__(self) -> bool:
        if False:
            while True:
                i = 10
        return bool(self._body)

class Section:

    def __init__(self, section_name: Optional[str]=None):
        if False:
            print('Hello World!')
        self._section_name = section_name

    @property
    def section_name(self) -> Optional[str]:
        if False:
            for i in range(10):
                print('nop')
        return self._section_name

class SimpleSection(Section):

    def __init__(self, section_name: str, value: Any=None):
        if False:
            while True:
                i = 10
        self._value = value
        super().__init__(section_name)

    @property
    def value(self) -> Any:
        if False:
            for i in range(10):
                print('nop')
        return self._value

    @value.setter
    def value(self, value: Any) -> None:
        if False:
            for i in range(10):
                print('nop')
        self._value = value

    def __bool__(self) -> bool:
        if False:
            while True:
                i = 10
        return bool(self._value)

class DictSection(Section, MutableMapping):

    def __init__(self, section_name: Optional[str]=None, items: Optional[List[SectionItem]]=None):
        if False:
            while True:
                i = 10
        self._items_dict = OrderedDict()
        if items:
            for item in items:
                self._items_dict[item.key] = item
        super().__init__(section_name)

    def copy(self) -> 'DictSection':
        if False:
            while True:
                i = 10
        return deepcopy(self)

    @property
    def section_items(self) -> List[SectionItem]:
        if False:
            while True:
                i = 10
        return list(self._items_dict.values())

    def __setitem__(self, k: str, v: Any) -> None:
        if False:
            print('Hello World!')
        if isinstance(v, DictSectionItem):
            self._items_dict[k] = v
        elif isinstance(v, Mapping):
            section_item_classes = {'Resources': Resource, 'Parameters': Parameter}
            class_name = self._section_name or ''
            item_class = section_item_classes.get(class_name, DictSectionItem)
            item = item_class(key=k, body=v)
            self._items_dict[k] = item
        else:
            self._items_dict[k] = SimpleSectionItem(key=k, value=v)

    def __delitem__(self, v: str) -> None:
        if False:
            return 10
        del self._items_dict[v]

    def __getitem__(self, k: str) -> Any:
        if False:
            for i in range(10):
                print('nop')
        v = self._items_dict[k]
        if isinstance(v, SimpleSectionItem):
            return v.value
        return v

    def __len__(self) -> int:
        if False:
            return 10
        return len(self._items_dict)

    def __iter__(self) -> Iterator:
        if False:
            print('Hello World!')
        return iter(self._items_dict)

    def __bool__(self) -> bool:
        if False:
            i = 10
            return i + 15
        return bool(self._items_dict)

class Resource(DictSectionItem):
    """
    Represents one resource in Resources section in a template
    """

    def __init__(self, key: Optional[str]=None, item_id: Optional[str]=None, body: Any=None, assets: Optional[List[Asset]]=None, nested_stack: Optional['Stack']=None, extra_details: Optional[Dict[str, Any]]=None):
        if False:
            while True:
                i = 10
        self._nested_stack = nested_stack
        super().__init__(key, item_id, body, assets, extra_details)

    def copy(self) -> 'Resource':
        if False:
            i = 10
            return i + 15
        return deepcopy(self)

    @property
    def nested_stack(self) -> Optional['Stack']:
        if False:
            print('Hello World!')
        return self._nested_stack

    @nested_stack.setter
    def nested_stack(self, nested_stack: 'Stack') -> None:
        if False:
            for i in range(10):
                print('nop')
        self._nested_stack = nested_stack

class Parameter(DictSectionItem):
    """
    Represents 1 Parameters in Parameters section in a template
    """

    def __init__(self, key: Optional[str]=None, item_id: Optional[str]=None, body: Any=None, added_by_iac: bool=False, assets: Optional[List[Asset]]=None, extra_details: Optional[Dict[str, Any]]=None):
        if False:
            print('Hello World!')
        self._added_by_iac = added_by_iac
        super().__init__(key, item_id, body, assets, extra_details)

    def copy(self) -> 'Parameter':
        if False:
            return 10
        return deepcopy(self)

    @property
    def added_by_iac(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        return self._added_by_iac

    @added_by_iac.setter
    def added_by_iac(self, added_by_iac: bool) -> None:
        if False:
            for i in range(10):
                print('nop')
        self._added_by_iac = added_by_iac

class Stack(MutableMapping):
    """
    Represents IaC Stack
    """

    def __init__(self, stack_id: Optional[str]=None, name: Optional[str]=None, origin_dir: Optional[str]=None, is_nested: bool=False, sections: Optional[Dict[str, Section]]=None, assets: Optional[List[Asset]]=None, environments: Optional[List[Environment]]=None, extra_details: Optional[Dict[str, Any]]=None):
        if False:
            print('Hello World!')
        self._stack_id = stack_id
        self._name = name
        self._is_nested = is_nested
        self._origin_dir = origin_dir or '.'
        if sections is None:
            sections = OrderedDict()
        self._sections = sections
        if assets is None:
            assets = []
        self._assets = assets
        if environments is None:
            environments = []
        self._environments = environments
        if extra_details is None:
            extra_details = {}
        self._extra_details = extra_details
        super().__init__()

    def copy(self) -> 'Stack':
        if False:
            for i in range(10):
                print('nop')
        return deepcopy(self)

    @property
    def stack_id(self) -> Optional[str]:
        if False:
            print('Hello World!')
        return self._stack_id or self.name

    @stack_id.setter
    def stack_id(self, stack_id: str) -> None:
        if False:
            while True:
                i = 10
        self._stack_id = stack_id

    @property
    def name(self) -> str:
        if False:
            return 10
        return self._name or ''

    @name.setter
    def name(self, name: str) -> None:
        if False:
            return 10
        self._name = name

    @property
    def origin_dir(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        return self._origin_dir

    @origin_dir.setter
    def origin_dir(self, origin_dir: str) -> None:
        if False:
            i = 10
            return i + 15
        self._origin_dir = origin_dir

    @property
    def is_nested(self) -> bool:
        if False:
            return 10
        return self._is_nested

    @is_nested.setter
    def is_nested(self, is_nested: bool) -> None:
        if False:
            while True:
                i = 10
        self._is_nested = is_nested

    @property
    def sections(self) -> Dict[str, Section]:
        if False:
            print('Hello World!')
        return self._sections

    @property
    def assets(self) -> List[Asset]:
        if False:
            while True:
                i = 10
        return self._assets

    @assets.setter
    def assets(self, assets: List[Asset]) -> None:
        if False:
            print('Hello World!')
        self._assets = assets

    @property
    def environments(self) -> Optional[List[Environment]]:
        if False:
            for i in range(10):
                print('nop')
        return self._environments

    @environments.setter
    def environments(self, environments: List[Environment]) -> None:
        if False:
            print('Hello World!')
        self._environments = environments

    @property
    def extra_details(self) -> Dict[str, Any]:
        if False:
            print('Hello World!')
        return self._extra_details

    @extra_details.setter
    def extra_details(self, extra_details: Dict[str, Any]) -> None:
        if False:
            return 10
        self._extra_details = extra_details

    def has_assets_of_package_type(self, package_type: str) -> bool:
        if False:
            print('Hello World!')
        package_type_to_asset_cls_map = {ZIP: S3Asset, IMAGE: ImageAsset}
        return any((isinstance(asset, package_type_to_asset_cls_map[package_type]) for asset in self.assets))

    def get_overrideable_parameters(self) -> Dict:
        if False:
            while True:
                i = 10
        '\n        Return a dict of parameters that are override-able, i.e. not added by iac\n        '
        return {key: val for (key, val) in self.get('Parameters', {}).items() if not val.added_by_iac}

    def as_dict(self) -> Union[MutableMapping, Mapping]:
        if False:
            for i in range(10):
                print('nop')
        '\n        return the stack as a dict for JSON serialization\n        '
        return _make_dict(self)

    def __setitem__(self, k: str, v: Any) -> None:
        if False:
            for i in range(10):
                print('nop')
        if isinstance(v, dict):
            section = DictSection(section_name=k)
            for key in v.keys():
                section[key] = v[key]
            self._sections[k] = section
        elif isinstance(v, Section):
            self._sections[k] = v
        else:
            self._sections[k] = SimpleSection(k, v)

    def __delitem__(self, v: str) -> None:
        if False:
            return 10
        del self._sections[v]

    def __getitem__(self, k: str) -> Any:
        if False:
            return 10
        v = self._sections[k]
        if isinstance(v, SimpleSection):
            return v.value
        return v

    def __len__(self) -> int:
        if False:
            return 10
        return len(self._sections)

    def __iter__(self) -> Iterator:
        if False:
            for i in range(10):
                print('nop')
        return iter(self._sections)

    def __bool__(self) -> bool:
        if False:
            print('Hello World!')
        return bool(self._sections)

class SamCliProject:
    """
    Class represents the Project data that will be returned by the IaC plugins
    Project:
        environments List[Environment]
        stacks List[Stack]
    """

    def __init__(self, stacks: List[Stack], extra_details: Optional[Dict[str, Any]]=None):
        if False:
            print('Hello World!')
        self._stacks = stacks or []
        self._extra_details = extra_details or {}

    @property
    def stacks(self) -> List[Stack]:
        if False:
            for i in range(10):
                print('nop')
        return self._stacks

    @stacks.setter
    def stacks(self, stacks: List[Stack]) -> None:
        if False:
            for i in range(10):
                print('nop')
        self._stacks = stacks

    @property
    def default_stack(self) -> Optional[Stack]:
        if False:
            for i in range(10):
                print('nop')
        if len(self._stacks) > 0:
            return self._stacks[0]
        return None

    @property
    def extra_details(self) -> Optional[Dict[str, Any]]:
        if False:
            return 10
        return self._extra_details

    @extra_details.setter
    def extra_details(self, extra_details: Dict[str, Any]) -> None:
        if False:
            i = 10
            return i + 15
        self._extra_details = extra_details

    def find_stack_by_name(self, name: str) -> Optional['Stack']:
        if False:
            for i in range(10):
                print('nop')
        for stack in self.stacks:
            if stack.name == name:
                return stack
        return None

class LookupPathType(Enum):
    SOURCE = 'Source'
    BUILD = 'BUILD'

class ProjectTypes(Enum):
    CFN = 'CFN'
    CDK = 'CDK'

class LookupPath:

    def __init__(self, lookup_path_dir: str, lookup_path_type: LookupPathType=LookupPathType.BUILD):
        if False:
            for i in range(10):
                print('nop')
        self._lookup_path_dir = lookup_path_dir
        self._lookup_path_type = lookup_path_type

    @property
    def lookup_path_dir(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        return self._lookup_path_dir

    @lookup_path_dir.setter
    def lookup_path_dir(self, lookup_path_dir: str) -> None:
        if False:
            print('Hello World!')
        self._lookup_path_dir = lookup_path_dir

    @property
    def lookup_path_type(self) -> LookupPathType:
        if False:
            i = 10
            return i + 15
        return self._lookup_path_type

    @lookup_path_type.setter
    def lookup_path_type(self, lookup_path_type: LookupPathType) -> None:
        if False:
            for i in range(10):
                print('nop')
        self._lookup_path_type = lookup_path_type

class SamCliContext:

    def __init__(self, command_options_map: Dict[str, Any], sam_command_name: str, is_guided: bool, is_debugging: bool, profile: Optional[Dict[str, Any]], region: Optional[str]):
        if False:
            i = 10
            return i + 15
        self._command_options_map = command_options_map
        self._sam_command_name = sam_command_name
        self._is_guided = is_guided
        self._is_debugging = is_debugging
        self._profile = profile
        self._region = region

    @property
    def command_options_map(self) -> Dict[str, Any]:
        if False:
            i = 10
            return i + 15
        '\n        the context retrieved from command line, its key is the command line option\n        name, value is corresponding input\n        '
        return self._command_options_map

    @property
    def sam_command_name(self) -> str:
        if False:
            while True:
                i = 10
        return self._sam_command_name

    @property
    def is_guided(self) -> bool:
        if False:
            print('Hello World!')
        return self._is_guided

    @property
    def is_debugging(self) -> bool:
        if False:
            print('Hello World!')
        return self._is_debugging

    @property
    def profile(self) -> Optional[Dict[str, Any]]:
        if False:
            i = 10
            return i + 15
        return self._profile

    @property
    def region(self) -> Optional[str]:
        if False:
            for i in range(10):
                print('nop')
        return self._region

class IaCPluginInterface(metaclass=abc.ABCMeta):
    """
    Interface for an IaC Plugin
    """

    def __init__(self, context: SamCliContext):
        if False:
            for i in range(10):
                print('nop')
        self._context = context

    @abc.abstractmethod
    def read_project(self, lookup_paths: List[LookupPath]) -> SamCliProject:
        if False:
            for i in range(10):
                print('nop')
        '\n        Read and parse template of that IaC Platform\n        '
        raise NotImplementedError

    @abc.abstractmethod
    def write_project(self, project: SamCliProject, build_dir: str) -> bool:
        if False:
            return 10
        "\n        Write project to a template (or a set of templates),\n        move the template(s) to build_path\n        return true if it's successful\n        "
        raise NotImplementedError

    @abc.abstractmethod
    def update_packaged_locations(self, stack: Stack) -> bool:
        if False:
            return 10
        "\n        update the locations of assets inside a stack after sam packaging\n        return true if it's successful\n        "
        raise NotImplementedError

    @staticmethod
    @abc.abstractmethod
    def get_iac_file_patterns() -> List[str]:
        if False:
            for i in range(10):
                print('nop')
        '\n        return a list of file types/patterns that define the IaC project\n        '
        raise NotImplementedError

def _make_dict(obj: Union[MutableMapping, Mapping]) -> Union[MutableMapping, Mapping]:
    if False:
        print('Hello World!')
    if not isinstance(obj, MutableMapping):
        return obj
    to_return = dict()
    for (key, val) in obj.items():
        to_return[key] = _make_dict(val)
    return to_return