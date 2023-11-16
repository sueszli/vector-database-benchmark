"""
Class that provides all nested stacks from a given SAM template
"""
import logging
import os
from typing import Dict, Iterator, List, Optional, Tuple, Union, cast
from urllib.parse import unquote, urlparse
from samcli.commands._utils.template import TemplateNotFoundException, get_template_data
from samcli.lib.providers.exceptions import RemoteStackLocationNotSupported
from samcli.lib.providers.provider import Stack, get_full_path
from samcli.lib.providers.sam_base_provider import SamBaseProvider
from samcli.lib.utils.resources import AWS_CLOUDFORMATION_STACK, AWS_SERVERLESS_APPLICATION
LOG = logging.getLogger(__name__)

class SamLocalStackProvider(SamBaseProvider):
    """
    Fetches and returns local nested stacks from a SAM Template. The SAM template passed to this provider is assumed
    to be valid, normalized and a dictionary.
    It may or may not contain a stack.
    """

    def __init__(self, template_file: str, stack_path: str, template_dict: Dict, parameter_overrides: Optional[Dict]=None, global_parameter_overrides: Optional[Dict]=None, use_sam_transform: bool=True):
        if False:
            i = 10
            return i + 15
        '\n        Initialize the class with SAM template data. The SAM template passed to this provider is assumed\n        to be valid and a dictionary. This class will perform template normalization to remove structures\n        like ``Globals``, resolve intrinsic functions etc.\n        This class does not perform any syntactic validation of the template.\n        After the class is initialized, any changes to the ``template_dict`` will not be reflected in here.\n        You need to explicitly update the class with new template, if necessary.\n        Parameters\n        ----------\n        template_file: str\n            SAM Stack Template file path\n        stack_path: str\n            SAM Stack stack_path (See samcli.lib.providers.provider.Stack.stack_path)\n        template_dict: dict\n            SAM Template as a dictionary\n        parameter_overrides: dict\n            Optional dictionary of values for SAM template parameters that might want to get substituted within\n            the template\n        global_parameter_overrides: dict\n            Optional dictionary of values for SAM template global parameters that might want to get substituted within\n            the template and all its child templates\n        use_sam_transform: bool\n            Whether to transform the given template with Serverless Application Model. Default is True\n        '
        self._template_file = template_file
        self._stack_path = stack_path
        self._template_dict = self.get_template(template_dict, SamLocalStackProvider.merge_parameter_overrides(parameter_overrides, global_parameter_overrides), use_sam_transform=use_sam_transform)
        self._resources = self._template_dict.get('Resources', {})
        self._global_parameter_overrides = global_parameter_overrides
        self._stacks: Dict[str, Stack] = {}
        self.remote_stack_full_paths: List[str] = []
        self._extract_stacks()
        LOG.debug('%d stacks found in the template', len(self._stacks))

    def get(self, name: str) -> Optional[Stack]:
        if False:
            while True:
                i = 10
        '\n        Returns the application given name or LogicalId of the application.\n        Every SAM resource has a logicalId, but it may\n        also have a application name. This method searches only for LogicalID and returns the application that matches\n        it.\n        :param string name: Name of the application\n        :return Function: namedtuple containing the Application information if application is found.\n                          None, if application is not found\n        :raises ValueError If name is not given\n        '
        for f in self.get_all():
            if f.name == name:
                return f
        return None

    def get_all(self) -> Iterator[Stack]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Yields all the applications available in the SAM Template.\n        :yields Application: map containing the application information\n        '
        for (_, stack) in self._stacks.items():
            yield stack

    def _extract_stacks(self) -> None:
        if False:
            return 10
        '\n        Extracts and returns nested application information from the given dictionary of SAM/CloudFormation resources.\n        This method supports applications defined with AWS::Serverless::Application\n        The dictionary of application LogicalId to the Application object will be assigned to self._stacks.\n        If child stacks with remote URL are detected, their full paths are recorded in self._remote_stack_full_paths.\n        '
        for (name, resource) in self._resources.items():
            resource_type = resource.get('Type')
            resource_properties = resource.get('Properties', {})
            resource_metadata = resource.get('Metadata', None)
            if resource_metadata:
                resource_properties['Metadata'] = resource_metadata
            stack: Optional[Stack] = None
            try:
                if resource_type == AWS_SERVERLESS_APPLICATION:
                    stack = SamLocalStackProvider._convert_sam_application_resource(self._template_file, self._stack_path, name, resource_properties)
                if resource_type == AWS_CLOUDFORMATION_STACK:
                    stack = SamLocalStackProvider._convert_cfn_stack_resource(self._template_file, self._stack_path, name, resource_properties)
            except RemoteStackLocationNotSupported:
                self.remote_stack_full_paths.append(get_full_path(self._stack_path, name))
            if stack:
                self._stacks[name] = stack

    @staticmethod
    def _convert_sam_application_resource(template_file: str, stack_path: str, name: str, resource_properties: Dict, global_parameter_overrides: Optional[Dict]=None) -> Optional[Stack]:
        if False:
            return 10
        location = resource_properties.get('Location')
        if isinstance(location, dict):
            raise RemoteStackLocationNotSupported()
        location = cast(str, location)
        if SamLocalStackProvider.is_remote_url(location):
            raise RemoteStackLocationNotSupported()
        if location.startswith('file://'):
            location = unquote(urlparse(location).path)
        else:
            location = SamLocalStackProvider.normalize_resource_path(template_file, location)
        return Stack(parent_stack_path=stack_path, name=name, location=location, parameters=SamLocalStackProvider.merge_parameter_overrides(resource_properties.get('Parameters', {}), global_parameter_overrides), template_dict=get_template_data(location), metadata=resource_properties.get('Metadata', {}))

    @staticmethod
    def _convert_cfn_stack_resource(template_file: str, stack_path: str, name: str, resource_properties: Dict, global_parameter_overrides: Optional[Dict]=None) -> Optional[Stack]:
        if False:
            while True:
                i = 10
        template_url = resource_properties.get('TemplateURL')
        if isinstance(template_url, dict):
            raise RemoteStackLocationNotSupported()
        template_url = cast(str, template_url)
        if SamLocalStackProvider.is_remote_url(template_url):
            raise RemoteStackLocationNotSupported()
        if template_url.startswith('file://'):
            template_url = unquote(urlparse(template_url).path)
        else:
            template_url = SamLocalStackProvider.normalize_resource_path(template_file, template_url)
        return Stack(parent_stack_path=stack_path, name=name, location=template_url, parameters=SamLocalStackProvider.merge_parameter_overrides(resource_properties.get('Parameters', {}), global_parameter_overrides), template_dict=get_template_data(template_url), metadata=resource_properties.get('Metadata', {}))

    @staticmethod
    def get_stacks(template_file: Optional[str]=None, stack_path: str='', name: str='', parameter_overrides: Optional[Dict]=None, global_parameter_overrides: Optional[Dict]=None, metadata: Optional[Dict]=None, template_dictionary: Optional[Dict]=None, use_sam_transform: bool=True) -> Tuple[List[Stack], List[str]]:
        if False:
            print('Hello World!')
        '\n        Recursively extract stacks from a template file.\n\n        Parameters\n        ----------\n        template_file: str\n            the file path of the template to extract stacks from. Only one of either template_dict or template_file\n            is required\n        stack_path: str\n            the stack path of the parent stack, for root stack, it is ""\n        name: str\n            the name of the stack associated with the template_file, for root stack, it is ""\n        parameter_overrides: Optional[Dict]\n            Optional dictionary of values for SAM template parameters that might want\n            to get substituted within the template\n        global_parameter_overrides: Optional[Dict]\n            Optional dictionary of values for SAM template global parameters\n            that might want to get substituted within the template and its child templates\n        metadata: Optional[Dict]\n            Optional dictionary of nested stack resource metadata values.\n        template_dictionary: Optional[Dict]\n            dictionary representing the sam template. Only one of either template_dict or template_file is required\n        use_sam_transform: bool\n            Whether to transform the given template with Serverless Application Model. Default is True\n\n        Returns\n        -------\n        stacks: List[Stack]\n            The list of stacks extracted from template_file\n        remote_stack_full_paths : List[str]\n            The list of full paths of detected remote stacks\n        '
        template_dict: dict
        if template_file:
            template_dict = get_template_data(template_file)
        elif template_dictionary:
            template_file = ''
            template_dict = template_dictionary
        else:
            raise TemplateNotFoundException(message='A template file or a template dict is required but both are missing.')
        stacks = [Stack(stack_path, name, template_file, SamLocalStackProvider.merge_parameter_overrides(parameter_overrides, global_parameter_overrides), template_dict, metadata)]
        remote_stack_full_paths: List[str] = []
        current = SamLocalStackProvider(template_file, stack_path, template_dict, parameter_overrides, global_parameter_overrides, use_sam_transform=use_sam_transform)
        remote_stack_full_paths.extend(current.remote_stack_full_paths)
        for child_stack in current.get_all():
            (stacks_in_child, remote_stack_full_paths_in_child) = SamLocalStackProvider.get_stacks(child_stack.location, os.path.join(stack_path, stacks[0].stack_id), child_stack.name, child_stack.parameters, global_parameter_overrides, child_stack.metadata, use_sam_transform=use_sam_transform)
            stacks.extend(stacks_in_child)
            remote_stack_full_paths.extend(remote_stack_full_paths_in_child)
        return (stacks, remote_stack_full_paths)

    @staticmethod
    def is_remote_url(url: str) -> bool:
        if False:
            print('Hello World!')
        return any([url.startswith(prefix) for prefix in ['s3://', 'http://', 'https://']])

    @staticmethod
    def find_root_stack(stacks: List[Stack]) -> Stack:
        if False:
            i = 10
            return i + 15
        candidates = [stack for stack in stacks if stack.is_root_stack]
        if not candidates:
            stacks_str = ', '.join([stack.stack_path for stack in stacks])
            raise ValueError(f'{stacks_str} does not contain a root stack')
        return candidates[0]

    @staticmethod
    def merge_parameter_overrides(parameter_overrides: Optional[Dict], global_parameter_overrides: Optional[Dict]) -> Dict:
        if False:
            i = 10
            return i + 15
        '\n        Combine global parameters and stack-specific parameters.\n        Right now the only global parameter override available is AWS::Region (via --region in "sam local"),\n        and AWS::Region won\'t appear in normal stack-specific parameter_overrides, so we don\'t\n        specify which type of parameters have high precedence.\n\n        Parameters\n        ----------\n        parameter_overrides: Optional[Dict]\n            stack-specific parameters\n        global_parameter_overrides: Optional[Dict]\n            global parameters\n\n        Returns\n        -------\n        Dict\n            merged dict containing both global and stack-specific parameters\n        '
        merged_parameter_overrides = {}
        merged_parameter_overrides.update(global_parameter_overrides or {})
        merged_parameter_overrides.update(parameter_overrides or {})
        return merged_parameter_overrides

    @staticmethod
    def normalize_resource_path(stack_file_path: str, path: str) -> str:
        if False:
            return 10
        '\n        Convert resource paths found in nested stack to ones resolvable from root stack.\n        For example,\n            root stack                -> template.yaml\n            child stack               -> folder/template.yaml\n            a resource in child stack -> folder/resource\n        the resource path is "resource" because it is extracted from child stack, the path is relative to child stack.\n        here we normalize the resource path into relative paths to root stack, which is "folder/resource"\n\n        * since stack_file_path might be a symlink, os.path.join() won\'t be able to derive the correct path.\n          for example, stack_file_path = \'folder/t.yaml\' -> \'../folder2/t.yaml\' and the path = \'src\'\n          the correct normalized path being returned should be \'../folder2/t.yaml\' but if we don\'t resolve the\n          symlink first, it would return \'folder/src.\'\n\n        * symlinks on Windows might not work properly.\n          https://stackoverflow.com/questions/43333640/python-os-path-realpath-for-symlink-in-windows\n          For example, using Python 3.7, realpath() is a no-op (same as abspath):\n            ```\n            Python 3.7.8 (tags/v3.7.8:4b47a5b6ba, Jun 28 2020, 08:53:46) [MSC v.1916 64 bit (AMD64)] on win32\n            Type "help", "copyright", "credits" or "license" for more information.\n            >>> import os\n            >>> os.symlink(\'some\\path\', \'link1\')\n            >>> os.path.realpath(\'link1\')\n            \'C:\\Users\\Administrator\\AppData\\Local\\Programs\\Python\\Python37\\link1\'\n            >>> os.path.islink(\'link1\')\n            True\n            ```\n          For Python 3.8, according to manual tests, 3.8.8 can resolve symlinks correctly while 3.8.0 cannot.\n\n\n        Parameters\n        ----------\n        stack_file_path\n            The file path of the stack containing the resource\n        path\n            the raw path read from the template dict\n\n        Returns\n        -------\n        str\n            the normalized path relative to root stack\n\n        '
        if os.path.isabs(path):
            return path
        if os.path.islink(stack_file_path):
            stack_file_path = os.path.relpath(os.path.realpath(stack_file_path))
        return os.path.normpath(os.path.join(os.path.dirname(stack_file_path), path))

def is_local_path(path: Union[Dict, str]) -> bool:
    if False:
        i = 10
        return i + 15
    return bool(path) and (not isinstance(path, dict)) and (not SamLocalStackProvider.is_remote_url(path))

def get_local_path(path: str, parent_path: str) -> str:
    if False:
        for i in range(10):
            print('nop')
    if path.startswith('file://'):
        path = unquote(urlparse(path).path)
    else:
        path = SamLocalStackProvider.normalize_resource_path(parent_path, path)
    return path