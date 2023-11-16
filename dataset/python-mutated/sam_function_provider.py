"""
Class that provides functions from a given SAM template
"""
import logging
from typing import Any, Dict, Iterator, List, Optional, cast
from samtranslator.policy_template_processor.exceptions import TemplateNotFoundException
from samcli.commands._utils.template import TemplateFailedParsingException
from samcli.commands.local.cli_common.user_exceptions import InvalidLayerVersionArn
from samcli.lib.providers.exceptions import InvalidLayerReference
from samcli.lib.utils.colors import Colored, Colors
from samcli.lib.utils.file_observer import FileObserver
from samcli.lib.utils.packagetype import IMAGE, ZIP
from samcli.lib.utils.resources import AWS_LAMBDA_FUNCTION, AWS_LAMBDA_LAYERVERSION, AWS_SERVERLESS_FUNCTION, AWS_SERVERLESS_LAYERVERSION
from ..build.constants import DEPRECATED_RUNTIMES
from .provider import Function, LayerVersion, Stack, get_full_path, get_function_build_info
from .sam_base_provider import SamBaseProvider
from .sam_stack_provider import SamLocalStackProvider
LOG = logging.getLogger(__name__)

class SamFunctionProvider(SamBaseProvider):
    """
    Fetches and returns Lambda Functions from a SAM Template. The SAM template passed to this provider is assumed
    to be valid, normalized and a dictionary.

    It may or may not contain a function.
    """

    def __init__(self, stacks: List[Stack], use_raw_codeuri: bool=False, ignore_code_extraction_warnings: bool=False, locate_layer_nested: bool=False) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Initialize the class with SAM template data. The SAM template passed to this provider is assumed\n        to be valid, normalized and a dictionary. It should be normalized by running all pre-processing\n        before passing to this class. The process of normalization will remove structures like ``Globals``, resolve\n        intrinsic functions etc.\n        This class does not perform any syntactic validation of the template.\n\n        After the class is initialized, any changes to the ``template_dict`` will not be reflected in here.\n        You need to explicitly update the class with new template, if necessary.\n\n        :param dict stacks: List of stacks functions are extracted from\n        :param bool use_raw_codeuri: Do not resolve adjust core_uri based on the template path, use the raw uri.\n            Note(xinhol): use_raw_codeuri is temporary to fix a bug, and will be removed for a permanent solution.\n        :param bool ignore_code_extraction_warnings: Ignores Log warnings\n        :param bool locate_layer_nested: resolved nested layer reference to their actual location in the nested stack\n        '
        self._stacks = stacks
        for stack in stacks:
            LOG.debug('%d resources found in the stack %s', len(stack.resources), stack.stack_path)
        self.functions = SamFunctionProvider._extract_functions(self._stacks, use_raw_codeuri, ignore_code_extraction_warnings, locate_layer_nested)
        self._colored = Colored()

    @property
    def stacks(self) -> List[Stack]:
        if False:
            return 10
        '\n        Returns the list of stacks (including the root stack and all children stacks)\n\n        :return list: list of stacks\n        '
        return self._stacks

    def update(self, stacks: List[Stack], use_raw_codeuri: bool=False, ignore_code_extraction_warnings: bool=False, locate_layer_nested: bool=False) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Hydrate the function provider with updated stacks\n        :param dict stacks: List of stacks functions are extracted from\n        :param bool use_raw_codeuri: Do not resolve adjust core_uri based on the template path, use the raw uri.\n            Note(xinhol): use_raw_codeuri is temporary to fix a bug, and will be removed for a permanent solution.\n        :param bool ignore_code_extraction_warnings: Ignores Log warnings\n        :param bool locate_layer_nested: resolved nested layer reference to their actual location in the nested stack\n        '
        self._stacks = stacks
        self.functions = SamFunctionProvider._extract_functions(self._stacks, use_raw_codeuri, ignore_code_extraction_warnings, locate_layer_nested)

    def get(self, name: str) -> Optional[Function]:
        if False:
            return 10
        '\n        Returns the function given name or LogicalId of the function. Every SAM resource has a logicalId, but it may\n        also have a function name. This method searches only for LogicalID and returns the function that matches.\n        If it is in a nested stack, "name" can be prefixed with stack path to avoid ambiguity.\n        For example, if a function with name "FunctionA" is located in StackN, which is a nested stack in root stack,\n          either "StackN/FunctionA" or "FunctionA" can be used.\n\n        :param string name: Name of the function\n        :return Function: namedtuple containing the Function information if function is found.\n                          None, if function is not found\n        :raises ValueError If name is not given\n        '
        if not name:
            raise ValueError('Function name is required')
        resolved_function = None
        if name in self.functions:
            resolved_function = self.functions.get(name)
        if not resolved_function:
            found_fs = []
            for f in self.get_all():
                if name in (f.function_id, f.name, f.functionname):
                    found_fs.append(f)
            if len(found_fs) > 1:
                found_fs.sort(key=lambda f0: f0.full_path.lower())
                message = f"Multiple functions found with keyword {name}! Function {found_fs[0].full_path} will be invoked! If it's not the function you are going to invoke, please choose one of them from below:"
                LOG.warning(Colored().yellow(message))
                for found_f in found_fs:
                    LOG.warning(Colored().yellow(found_f.full_path))
                resolved_function = found_fs[0]
            elif len(found_fs) == 1:
                resolved_function = found_fs[0]
        if resolved_function:
            self._deprecate_notification(resolved_function.runtime)
        return resolved_function

    def _deprecate_notification(self, runtime: Optional[str]) -> None:
        if False:
            print('Hello World!')
        if runtime in DEPRECATED_RUNTIMES:
            message = f'WARNING: {runtime} is no longer supported by AWS Lambda, please update to a newer supported runtime. For more information please check AWS Lambda Runtime Support Policy: https://docs.aws.amazon.com/lambda/latest/dg/runtime-support-policy.html'
            LOG.warning(self._colored.color_log(msg=message, color=Colors.WARNING), extra=dict(markup=True))

    def get_all(self) -> Iterator[Function]:
        if False:
            i = 10
            return i + 15
        '\n        Yields all the Lambda functions available in the SAM Template.\n\n        :yields Function: namedtuple containing the function information\n        '
        for (_, function) in self.functions.items():
            yield function

    @staticmethod
    def _extract_functions(stacks: List[Stack], use_raw_codeuri: bool=False, ignore_code_extraction_warnings: bool=False, locate_layer_nested: bool=False) -> Dict[str, Function]:
        if False:
            i = 10
            return i + 15
        '\n        Extracts and returns function information from the given dictionary of SAM/CloudFormation resources. This\n        method supports functions defined with AWS::Serverless::Function and AWS::Lambda::Function\n\n        :param stacks: List of SAM/CloudFormation stacks to extract functions from\n        :param bool use_raw_codeuri: Do not resolve adjust core_uri based on the template path, use the raw uri.\n        :param bool ignore_code_extraction_warnings: suppress log statements on code extraction from resources.\n        :param bool locate_layer_nested: resolved nested layer reference to their actual location in the nested stack\n        :return dict(string : samcli.commands.local.lib.provider.Function): Dictionary of function full_path to the\n            Function configuration object\n        '
        result: Dict[str, Function] = {}
        for stack in stacks:
            for (name, resource) in stack.resources.items():
                resource_type = resource.get('Type')
                resource_properties = resource.get('Properties', {})
                resource_metadata = resource.get('Metadata', None)
                if resource_metadata:
                    resource_properties['Metadata'] = resource_metadata
                if resource_type in [AWS_SERVERLESS_FUNCTION, AWS_LAMBDA_FUNCTION]:
                    resource_package_type = resource_properties.get('PackageType', ZIP)
                    code_property_key = SamBaseProvider.CODE_PROPERTY_KEYS[resource_type]
                    image_property_key = SamBaseProvider.IMAGE_PROPERTY_KEYS[resource_type]
                    if resource_package_type == ZIP and SamBaseProvider._is_s3_location(resource_properties.get(code_property_key)):
                        if not ignore_code_extraction_warnings:
                            SamFunctionProvider._warn_code_extraction(resource_type, name, code_property_key)
                        continue
                    if resource_package_type == IMAGE and SamBaseProvider._is_ecr_uri(resource_properties.get(image_property_key)) and (not SamFunctionProvider._metadata_has_necessary_entries_for_image_function_to_be_built(resource_metadata)):
                        if not ignore_code_extraction_warnings:
                            SamFunctionProvider._warn_imageuri_extraction(resource_type, name, image_property_key)
                        continue
                if resource_type == AWS_SERVERLESS_FUNCTION:
                    layers = SamFunctionProvider._parse_layer_info(stack, resource_properties.get('Layers', []), use_raw_codeuri, ignore_code_extraction_warnings=ignore_code_extraction_warnings, locate_layer_nested=locate_layer_nested, stacks=stacks if locate_layer_nested else None, function_id=resource_metadata.get('SamResourceId', '') if locate_layer_nested else None)
                    function = SamFunctionProvider._convert_sam_function_resource(stack, name, resource_properties, layers, use_raw_codeuri)
                    result[function.full_path] = function
                elif resource_type == AWS_LAMBDA_FUNCTION:
                    layers = SamFunctionProvider._parse_layer_info(stack, resource_properties.get('Layers', []), use_raw_codeuri, ignore_code_extraction_warnings=ignore_code_extraction_warnings, locate_layer_nested=locate_layer_nested, stacks=stacks if locate_layer_nested else None, function_id=resource_metadata.get('SamResourceId', '') if locate_layer_nested else None)
                    function = SamFunctionProvider._convert_lambda_function_resource(stack, name, resource_properties, layers, use_raw_codeuri)
                    result[function.full_path] = function
        return result

    @staticmethod
    def _convert_sam_function_resource(stack: Stack, name: str, resource_properties: Dict, layers: List[LayerVersion], use_raw_codeuri: bool=False) -> Function:
        if False:
            return 10
        '\n        Converts a AWS::Serverless::Function resource to a Function configuration usable by the provider.\n\n        Parameters\n        ----------\n        name str\n            LogicalID of the resource NOTE: This is *not* the function name because not all functions declare a name\n        resource_properties dict\n            Properties of this resource\n        layers List(samcli.commands.local.lib.provider.Layer)\n            List of the Layer objects created from the template and layer list defined on the function.\n\n        Returns\n        -------\n        samcli.commands.local.lib.provider.Function\n            Function configuration\n        '
        codeuri: Optional[str] = SamFunctionProvider.DEFAULT_CODEURI
        inlinecode = resource_properties.get('InlineCode')
        imageuri = None
        function_id = SamFunctionProvider._get_function_id(resource_properties, name)
        packagetype = resource_properties.get('PackageType', ZIP)
        if packagetype == ZIP:
            if inlinecode:
                LOG.debug("Found Serverless function with name='%s' and InlineCode", name)
                codeuri = None
            else:
                codeuri = SamBaseProvider._extract_codeuri(resource_properties, 'CodeUri')
                LOG.debug("Found Serverless function with name='%s' and CodeUri='%s'", name, codeuri)
        elif packagetype == IMAGE:
            imageuri = SamFunctionProvider._extract_sam_function_imageuri(resource_properties, 'ImageUri')
            LOG.debug("Found Serverless function with name='%s' and ImageUri='%s'", name, imageuri)
        return SamFunctionProvider._build_function_configuration(stack, function_id, name, codeuri, resource_properties, layers, inlinecode, imageuri, use_raw_codeuri)

    @staticmethod
    def _get_function_id(resource_properties: Dict, logical_id: str) -> str:
        if False:
            print('Hello World!')
        '\n        Get unique id for Function resource.\n        For CFN/SAM project, this function id is the logical id.\n        For CDK project, this function id is the user-defined resource id, or the logical id if the resource id is not\n        found.\n\n        Parameters\n        ----------\n        resource_properties str\n            Properties of this resource\n        logical_id str\n            LogicalID of the resource\n\n        Returns\n        -------\n        str\n            The unique function id\n        '
        function_id = resource_properties.get('Metadata', {}).get('SamResourceId')
        if isinstance(function_id, str) and function_id:
            return function_id
        return logical_id

    @staticmethod
    def _convert_lambda_function_resource(stack: Stack, name: str, resource_properties: Dict, layers: List[LayerVersion], use_raw_codeuri: bool=False) -> Function:
        if False:
            return 10
        '\n        Converts a AWS::Lambda::Function resource to a Function configuration usable by the provider.\n\n        Parameters\n        ----------\n        name str\n            LogicalID of the resource NOTE: This is *not* the function name because not all functions declare a name\n        resource_properties dict\n            Properties of this resource\n        layers List(samcli.commands.local.lib.provider.Layer)\n            List of the Layer objects created from the template and layer list defined on the function.\n        use_raw_codeuri\n            Do not resolve adjust core_uri based on the template path, use the raw uri.\n\n        Returns\n        -------\n        samcli.commands.local.lib.provider.Function\n            Function configuration\n        '
        codeuri: Optional[str] = SamFunctionProvider.DEFAULT_CODEURI
        inlinecode = None
        imageuri = None
        function_id = SamFunctionProvider._get_function_id(resource_properties, name)
        packagetype = resource_properties.get('PackageType', ZIP)
        if packagetype == ZIP:
            if 'Code' in resource_properties and isinstance(resource_properties['Code'], dict) and resource_properties['Code'].get('ZipFile'):
                inlinecode = resource_properties['Code']['ZipFile']
                LOG.debug("Found Lambda function with name='%s' and Code ZipFile", name)
                codeuri = None
            else:
                codeuri = SamBaseProvider._extract_codeuri(resource_properties, 'Code')
                LOG.debug("Found Lambda function with name='%s' and CodeUri='%s'", name, codeuri)
        elif packagetype == IMAGE:
            imageuri = SamFunctionProvider._extract_lambda_function_imageuri(resource_properties, 'Code')
            LOG.debug("Found Lambda function with name='%s' and Imageuri='%s'", name, imageuri)
        return SamFunctionProvider._build_function_configuration(stack, function_id, name, codeuri, resource_properties, layers, inlinecode, imageuri, use_raw_codeuri)

    @staticmethod
    def _build_function_configuration(stack: Stack, function_id: str, name: str, codeuri: Optional[str], resource_properties: Dict, layers: List, inlinecode: Optional[str], imageuri: Optional[str], use_raw_codeuri: bool=False) -> Function:
        if False:
            while True:
                i = 10
        '\n        Builds a Function configuration usable by the provider.\n\n        Parameters\n        ----------\n        name str\n            LogicalID of the resource NOTE: This is *not* the function name because not all functions declare a name\n        function_id str\n            Unique function id\n        codeuri str\n            Representing the local code path\n        resource_properties dict\n            Properties of this resource\n        layers List(samcli.commands.local.lib.provider.Layer)\n            List of the Layer objects created from the template and layer list defined on the function.\n        use_raw_codeuri\n            Do not resolve adjust core_uri based on the template path, use the raw uri.\n\n        Returns\n        -------\n        samcli.commands.local.lib.provider.Function\n            Function configuration\n        '
        metadata = resource_properties.get('Metadata', None)
        if metadata and 'DockerContext' in metadata and (not use_raw_codeuri):
            LOG.debug('--base-dir is not presented, adjusting uri %s relative to %s', metadata['DockerContext'], stack.location)
            metadata['DockerContext'] = SamLocalStackProvider.normalize_resource_path(stack.location, metadata['DockerContext'])
        if codeuri and (not use_raw_codeuri):
            LOG.debug('--base-dir is not presented, adjusting uri %s relative to %s', codeuri, stack.location)
            codeuri = SamLocalStackProvider.normalize_resource_path(stack.location, codeuri)
        package_type = resource_properties.get('PackageType', ZIP)
        function_build_info = get_function_build_info(get_full_path(stack.stack_path, function_id), package_type, inlinecode, codeuri, metadata)
        return Function(stack_path=stack.stack_path, function_id=function_id, name=name, functionname=resource_properties.get('FunctionName', name), packagetype=package_type, runtime=resource_properties.get('Runtime'), memory=resource_properties.get('MemorySize'), timeout=resource_properties.get('Timeout'), handler=resource_properties.get('Handler'), codeuri=codeuri, imageuri=imageuri if imageuri else resource_properties.get('ImageUri'), imageconfig=resource_properties.get('ImageConfig'), environment=resource_properties.get('Environment'), rolearn=resource_properties.get('Role'), events=resource_properties.get('Events'), layers=layers, metadata=metadata, inlinecode=inlinecode, codesign_config_arn=resource_properties.get('CodeSigningConfigArn', None), architectures=resource_properties.get('Architectures', None), function_url_config=resource_properties.get('FunctionUrlConfig'), runtime_management_config=resource_properties.get('RuntimeManagementConfig'), function_build_info=function_build_info)

    @staticmethod
    def _parse_layer_info(stack: Stack, list_of_layers: List[Any], use_raw_codeuri: bool=False, ignore_code_extraction_warnings: bool=False, locate_layer_nested: bool=False, stacks: Optional[List[Stack]]=None, function_id: Optional[str]=None) -> List[LayerVersion]:
        if False:
            while True:
                i = 10
        '\n        Creates a list of Layer objects that are represented by the resources and the list of layers\n\n        Parameters\n        ----------\n        stack : Stack\n            The stack the layer is defined in\n        list_of_layers : List[Any]\n            List of layers that are defined within the Layers Property on a function,\n            layer can be defined as string or Dict, in case customers define it in other types, use "Any" here.\n        use_raw_codeuri : bool\n            Do not resolve adjust core_uri based on the template path, use the raw uri.\n        ignore_code_extraction_warnings : bool\n            Whether to print warning when codeuri is not a local pth\n        locate_layer_nested: bool\n            Resolved nested layer reference to their actual location in the nested stack\n        stacks: List[Stack]\n            List of stacks generates from templates\n        function_id: str\n            Logical id for the function resources\n\n        Returns\n        -------\n        List(samcli.commands.local.lib.provider.Layer)\n            List of the Layer objects created from the template and layer list defined on the function. The order\n            of the layers does not change.\n\n            I.E: list_of_layers = ["layer1", "layer2"] the return would be [Layer("layer1"), Layer("layer2")]\n        '
        layers = []
        if locate_layer_nested and stacks and function_id:
            func_template = stack.template_dict.get('Resources', {}).get(function_id, {})
            a_list_of_layers = func_template.get('Properties', {}).get('Layers', [])
            for layer in a_list_of_layers:
                found_layer = SamFunctionProvider._locate_layer_from_nested(stack, stacks, layer, use_raw_codeuri, ignore_code_extraction_warnings)
                if found_layer:
                    layers.append(found_layer)
        for layer in list_of_layers:
            if layer == 'arn:aws:lambda:::awslayer:AmazonLinux1803':
                LOG.debug('Skipped arn:aws:lambda:::awslayer:AmazonLinux1803 as the containers are AmazonLinux1803')
                continue
            if layer == 'arn:aws:lambda:::awslayer:AmazonLinux1703':
                raise InvalidLayerVersionArn('Building and invoking locally only supports AmazonLinux1803. See https://aws.amazon.com/blogs/compute/upcoming-updates-to-the-aws-lambda-execution-environment/ for more detials.')
            if isinstance(layer, str):
                if locate_layer_nested and 'arn:' not in layer:
                    continue
                layers.append(LayerVersion(layer, None, stack_path=stack.stack_path))
                continue
            if isinstance(layer, dict) and layer.get('Ref'):
                if locate_layer_nested:
                    continue
                found_layer = SamFunctionProvider._locate_layer_from_ref(stack, layer, use_raw_codeuri, ignore_code_extraction_warnings)
                if found_layer:
                    layers.append(found_layer)
            else:
                LOG.debug('layer "%s" is not recognizable, it might be using intrinsic functions that we don\'t support yet. Skipping.', str(layer))
        return layers

    @staticmethod
    def _locate_layer_from_nested(stack: Stack, stacks: List[Stack], layer: Any, use_raw_codeuri: bool=False, ignore_code_extraction_warnings: bool=False) -> Optional[LayerVersion]:
        if False:
            print('Hello World!')
        '\n        Search the layer reference through all the local templates and try to find it\'s actual location then create a\n        layer object and return\n\n        Right now this method does not support more complicated intrinsics like Fn:Sub and Fn:If, future task to\n        expand support. One of possible solutions can be adding in an intrinsic resovler.\n\n        TODO: this function have too many return statemnets, we may need to refactor it, break it down to multiple\n        sub functions for example.\n\n        Parameters\n        ----------\n        stack : Stack\n            The stack the layer is defined in\n        stacks: List[Stack]\n            List of stacks generates from templates\n        layer : Any\n            layer that are defined within the Layers Property on a function,\n            layer can be defined as string or Dict, in case customers define it in other types, use "Any" here.\n        use_raw_codeuri : bool\n            Do not resolve adjust core_uri based on the template path, use the raw uri.\n        ignore_code_extraction_warnings : bool\n            Whether to print warning when codeuri is not a local path\n\n        Returns\n        -------\n        samcli.commands.local.lib.provider.Layer\n            The Layer object created from the template and layer defined on the function.\n        '
        if isinstance(layer, str):
            outputs = stack.template_dict.get('Outputs', {})
            LOG.debug("Search layer %s in %s 's Output section", layer, stack.stack_path)
            if layer not in outputs:
                LOG.debug('Layer not in Output section, layer can not be located in templates')
                return None
            layer = outputs.get(layer).get('Value')
            LOG.debug('Layer found in Output section, try to search it in current stack %s', stack.stack_path)
        layer_reference = None
        if isinstance(layer, dict) and layer.get('Ref'):
            layer_reference = layer.get('Ref')
        elif isinstance(layer, dict) and layer.get('Fn::GetAtt'):
            layer_attribute: List = layer.get('Fn::GetAtt', [])
            if not SamFunctionProvider._validate_layer_get_attr_format(layer):
                return None
            layer_stack_reference = layer_attribute[0]
            layer_reference = layer_attribute[1].split('.')[1]
            LOG.debug('Search layer %s in child stack', layer_reference)
            child_stacks = Stack.get_child_stacks(stack, stacks)
            stack_prefix = stack.stack_path + '/' if stack.stack_path else ''
            stack_path = stack_prefix + layer_stack_reference
            child_stack = Stack.get_stack_by_full_path(stack_path, child_stacks)
            if not child_stack:
                LOG.debug('Child stack not found, layer can not be located in templates')
                return None
            LOG.debug('Child stack %s found', child_stack.stack_path)
            return SamFunctionProvider._locate_layer_from_nested(child_stack, stacks, layer_reference, use_raw_codeuri, ignore_code_extraction_warnings)
        parameters: Dict = stack.template_dict.get('Parameters', {})
        if not parameters or (layer_reference and layer_reference not in parameters):
            LOG.debug('Resolved layer: %s in current stack %s', layer_reference, stack.stack_path)
            try:
                resolve_layer = SamFunctionProvider._locate_layer_from_ref(stack, layer, use_raw_codeuri, ignore_code_extraction_warnings)
            except InvalidLayerReference:
                LOG.debug("Layer reference (%s) can't be located in the template", layer)
                return None
            return resolve_layer
        parent_stack = Stack.get_parent_stack(stack, stacks)
        LOG.debug('Search layer: %s in parent stack', layer_reference)
        if not parent_stack:
            LOG.debug('Parent stack not found, layer can not be located in templates')
            return None
        LOG.debug('Found parent stack: %s', parent_stack.stack_path)
        layer = parent_stack.template_dict.get('Resources', {}).get(stack.name, {}).get('Properties', {}).get('Parameters', {}).get(layer_reference)
        return SamFunctionProvider._locate_layer_from_nested(parent_stack, stacks, layer, use_raw_codeuri, ignore_code_extraction_warnings)

    @staticmethod
    def _validate_layer_get_attr_format(layer: Dict) -> bool:
        if False:
            return 10
        warn_message = 'Fn::GetAtt with unsupported format in accelerate nested stack'
        layer_attribute = layer.get('Fn::GetAtt', [])
        required_layer_attr_length = 2
        reqauired_layer_reference_length = 2
        if not isinstance(layer_attribute, List):
            LOG.warning(warn_message)
            return False
        if len(layer_attribute) != required_layer_attr_length:
            LOG.warning(warn_message)
            return False
        layer_reference_array = layer_attribute[1].split('.')
        if len(layer_reference_array) != reqauired_layer_reference_length:
            LOG.warning(warn_message)
            return False
        return True

    @staticmethod
    def _locate_layer_from_ref(stack: Stack, layer: Dict, use_raw_codeuri: bool=False, ignore_code_extraction_warnings: bool=False) -> Optional[LayerVersion]:
        if False:
            print('Hello World!')
        layer_logical_id = cast(str, layer.get('Ref'))
        layer_resource = stack.resources.get(layer_logical_id)
        if not layer_resource or layer_resource.get('Type', '') not in (AWS_SERVERLESS_LAYERVERSION, AWS_LAMBDA_LAYERVERSION):
            raise InvalidLayerReference()
        layer_properties = layer_resource.get('Properties', {})
        resource_type = layer_resource.get('Type')
        compatible_runtimes = layer_properties.get('CompatibleRuntimes')
        codeuri: Optional[str] = None
        if resource_type in [AWS_LAMBDA_LAYERVERSION, AWS_SERVERLESS_LAYERVERSION]:
            code_property_key = SamBaseProvider.CODE_PROPERTY_KEYS[resource_type]
            if SamBaseProvider._is_s3_location(layer_properties.get(code_property_key)):
                if not ignore_code_extraction_warnings:
                    SamFunctionProvider._warn_code_extraction(resource_type, layer_logical_id, code_property_key)
                return None
            codeuri = SamBaseProvider._extract_codeuri(layer_properties, code_property_key)
        if codeuri and (not use_raw_codeuri):
            LOG.debug('--base-dir is not presented, adjusting uri %s relative to %s', codeuri, stack.location)
            codeuri = SamLocalStackProvider.normalize_resource_path(stack.location, codeuri)
        return LayerVersion(layer_logical_id, codeuri, compatible_runtimes, layer_resource.get('Metadata', None), stack_path=stack.stack_path)

    def get_resources_by_stack_path(self, stack_path: str) -> Dict:
        if False:
            for i in range(10):
                print('nop')
        candidates = [stack.resources for stack in self._stacks if stack.stack_path == stack_path]
        if not candidates:
            raise RuntimeError(f'Cannot find resources with stack_path = {stack_path}')
        return candidates[0]

    @staticmethod
    def _metadata_has_necessary_entries_for_image_function_to_be_built(metadata: Optional[Dict[str, Any]]) -> bool:
        if False:
            while True:
                i = 10
        '\n        > Note: If the PackageType property is set to Image, then either ImageUri is required,\n          or you must build your application with necessary Metadata entries in the AWS SAM template file.\n          https://docs.aws.amazon.com/serverless-application-model/latest/developerguide/sam-resource-function.html#sam-function-imageuri\n\n        When ImageUri and Metadata are both provided, we will try to determine whether to treat the function\n        as to be built or to be skipped. When we skip it whenever "ImageUri" is provided,\n        we introduced a breaking change https://github.com/aws/aws-sam-cli/issues/3239\n\n        This function is used to check whether there are the customers have "intention" to\n        let AWS SAM CLI to build this image function.\n        '
        return isinstance(metadata, dict) and bool(metadata.get('DockerContext'))

class RefreshableSamFunctionProvider(SamFunctionProvider):
    """
    Fetches and returns Lambda Functions from a SAM Template. The SAM template passed to this provider is assumed
    to be valid, normalized and a dictionary. It also detects any stack template change, and refreshes the loaded
    functions.

    It may or may not contain a function.
    """

    def __init__(self, stacks: List[Stack], parameter_overrides: Optional[Dict]=None, global_parameter_overrides: Optional[Dict]=None, use_raw_codeuri: bool=False, ignore_code_extraction_warnings: bool=False) -> None:
        if False:
            while True:
                i = 10
        '\n        Initialize the class with SAM template data. The SAM template passed to this provider is assumed\n        to be valid, normalized and a dictionary. It should be normalized by running all pre-processing\n        before passing to this class. The process of normalization will remove structures like ``Globals``, resolve\n        intrinsic functions etc.\n        This class does not perform any syntactic validation of the template.\n\n        This Class will also initialize watchers, to check the stack templates for any update, and refresh the loaded\n        functions.\n\n        :param dict stacks: List of stacks functions are extracted from\n        :param bool use_raw_codeuri: Do not resolve adjust core_uri based on the template path, use the raw uri.\n            Note(xinhol): use_raw_codeuri is temporary to fix a bug, and will be removed for a permanent solution.\n        :param bool ignore_code_extraction_warnings: Ignores Log warnings\n        '
        super().__init__(stacks, use_raw_codeuri, ignore_code_extraction_warnings)
        self._use_raw_codeuri = use_raw_codeuri
        self._ignore_code_extraction_warnings = ignore_code_extraction_warnings
        self._parameter_overrides = parameter_overrides
        self._global_parameter_overrides = global_parameter_overrides
        self.parent_templates_paths = []
        for stack in self._stacks:
            if stack.is_root_stack:
                self.parent_templates_paths.append(stack.location)
        self.is_changed = False
        self._observer = FileObserver(self._set_templates_changed)
        self._observer.start()
        self._watch_stack_templates(stacks)

    @property
    def stacks(self) -> List[Stack]:
        if False:
            print('Hello World!')
        '\n        It Checks if any template got changed, then refresh the loaded stacks, and functions.\n\n        Returns the list of stacks (including the root stack and all children stacks)\n\n        :return list: list of stacks\n        '
        if self.is_changed:
            self._refresh_loaded_functions()
        return super().stacks

    def get(self, name: str) -> Optional[Function]:
        if False:
            i = 10
            return i + 15
        '\n        It Checks if any template got changed, then refresh the loaded functions before finding the required function.\n\n        Returns the function given name or LogicalId of the function. Every SAM resource has a logicalId, but it may\n        also have a function name. This method searches only for LogicalID and returns the function that matches.\n        If it is in a nested stack, "name" can be prefixed with stack path to avoid ambiguity.\n        For example, if a function with name "FunctionA" is located in StackN, which is a nested stack in root stack,\n          either "StackN/FunctionA" or "FunctionA" can be used.\n\n        :param string name: Name of the function\n        :return Function: namedtuple containing the Function information if function is found.\n                          None, if function is not found\n        :raises ValueError If name is not given\n        '
        if self.is_changed:
            self._refresh_loaded_functions()
        return super().get(name)

    def get_all(self) -> Iterator[Function]:
        if False:
            while True:
                i = 10
        '\n        It Checks if any template got changed, then refresh the loaded functions before returning all available\n        functions.\n\n        Yields all the Lambda functions available in the SAM Template.\n\n        :yields Function: namedtuple containing the function information\n        '
        if self.is_changed:
            self._refresh_loaded_functions()
        return super().get_all()

    def get_resources_by_stack_path(self, stack_path: str) -> Dict:
        if False:
            while True:
                i = 10
        if self.is_changed:
            self._refresh_loaded_functions()
        return super().get_resources_by_stack_path(stack_path)

    def _set_templates_changed(self, paths: List[str]) -> None:
        if False:
            for i in range(10):
                print('nop')
        LOG.info('A change got detected in the templates %s. Mark templates as changed to be reloaded in the next invoke', ', '.join(paths))
        self.is_changed = True
        for stack in self._stacks:
            self._observer.unwatch(stack.location)

    def _watch_stack_templates(self, stacks: List[Stack]) -> None:
        if False:
            i = 10
            return i + 15
        '\n        initialize the list of stack template watchers\n        '
        for stack in stacks:
            self._observer.watch(stack.location)

    def _refresh_loaded_functions(self) -> None:
        if False:
            print('Hello World!')
        '\n        Reload the stacks, and lambda functions from template files.\n        '
        LOG.debug('A change got detected in one of the stack templates. Reload the lambda function resources')
        self._stacks = []
        for template_file in self.parent_templates_paths:
            try:
                (template_stacks, _) = SamLocalStackProvider.get_stacks(template_file, parameter_overrides=self._parameter_overrides, global_parameter_overrides=self._global_parameter_overrides)
                self._stacks += template_stacks
            except (TemplateNotFoundException, TemplateFailedParsingException) as ex:
                raise ex
        self.is_changed = False
        self.functions = self._extract_functions(self._stacks, self._use_raw_codeuri, self._ignore_code_extraction_warnings)
        self._watch_stack_templates(self._stacks)

    def stop_observer(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Stop Observing.\n        '
        self._observer.stop()