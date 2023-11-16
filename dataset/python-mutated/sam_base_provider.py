"""
Base class for SAM Template providers
"""
import logging
from typing import Any, Dict, Iterable, Optional, Union, cast
from samcli.lib.iac.plugins_interfaces import Stack
from samcli.lib.intrinsic_resolver.intrinsic_property_resolver import IntrinsicResolver
from samcli.lib.intrinsic_resolver.intrinsics_symbol_table import IntrinsicsSymbolTable
from samcli.lib.package.ecr_utils import is_ecr_url
from samcli.lib.samlib.resource_metadata_normalizer import ResourceMetadataNormalizer
from samcli.lib.samlib.wrapper import SamTranslatorWrapper
from samcli.lib.utils.resources import AWS_LAMBDA_FUNCTION, AWS_LAMBDA_LAYERVERSION, AWS_SERVERLESS_FUNCTION, AWS_SERVERLESS_LAYERVERSION
LOG = logging.getLogger(__name__)

class SamBaseProvider:
    """
    Base class for SAM Template providers
    """
    DEFAULT_CODEURI = '.'
    CODE_PROPERTY_KEYS = {AWS_LAMBDA_FUNCTION: 'Code', AWS_SERVERLESS_FUNCTION: 'CodeUri', AWS_LAMBDA_LAYERVERSION: 'Content', AWS_SERVERLESS_LAYERVERSION: 'ContentUri'}
    IMAGE_PROPERTY_KEYS = {AWS_LAMBDA_FUNCTION: 'Code', AWS_SERVERLESS_FUNCTION: 'ImageUri'}

    def get(self, name: str) -> Optional[Any]:
        if False:
            print('Hello World!')
        '\n        Given name of the function, this method must return the Function object\n\n        :param string name: Name of the function\n        :return Function: namedtuple containing the Function information\n        '
        raise NotImplementedError('not implemented')

    def get_all(self) -> Iterable:
        if False:
            i = 10
            return i + 15
        '\n        Yields all the Lambda functions available in the provider.\n\n        :yields Function: namedtuple containing the function information\n        '
        raise NotImplementedError('not implemented')

    @staticmethod
    def _extract_codeuri(resource_properties: Dict, code_property_key: str) -> str:
        if False:
            print('Hello World!')
        '\n        Extracts the Function/Layer code path from the Resource Properties\n\n        Parameters\n        ----------\n        resource_properties dict\n            Dictionary representing the Properties of the Resource\n        code_property_key str\n            Property Key of the code on the Resource\n\n        Returns\n        -------\n        str\n            Representing the local code path\n        '
        codeuri = resource_properties.get(code_property_key, SamBaseProvider.DEFAULT_CODEURI)
        if isinstance(codeuri, dict):
            return SamBaseProvider.DEFAULT_CODEURI
        return cast(str, codeuri)

    @staticmethod
    def _is_s3_location(location: Optional[Union[str, Dict]]) -> bool:
        if False:
            print('Hello World!')
        '\n        the input could be:\n        - CodeUri of Serverless::Function\n        - Code of Lambda::Function\n        - ContentUri of Serverless::LayerVersion\n        - Content of Lambda::LayerVersion\n        '
        return isinstance(location, dict) and ('S3Bucket' in location or 'Bucket' in location) or (isinstance(location, str) and location.startswith('s3://'))

    @staticmethod
    def _is_ecr_uri(location: Optional[Union[str, Dict]]) -> bool:
        if False:
            return 10
        '\n        the input could be:\n        - ImageUri of Serverless::Function\n        - Code of Lambda::Function\n        '
        return location is not None and is_ecr_url(str(location.get('ImageUri', '')) if isinstance(location, dict) else location)

    @staticmethod
    def _warn_code_extraction(resource_type: str, resource_name: str, code_property: str) -> None:
        if False:
            return 10
        LOG.warning("The resource %s '%s' has specified S3 location for %s. It will not be built and SAM CLI does not support invoking it locally.", resource_type, resource_name, code_property)

    @staticmethod
    def _warn_imageuri_extraction(resource_type: str, resource_name: str, image_property: str) -> None:
        if False:
            i = 10
            return i + 15
        LOG.warning("The resource %s '%s' has specified ECR registry image for %s. It will not be built and SAM CLI does not support invoking it locally.", resource_type, resource_name, image_property)

    @staticmethod
    def _extract_lambda_function_imageuri(resource_properties: Dict, code_property_key: str) -> Optional[str]:
        if False:
            print('Hello World!')
        '\n        Extracts the Lambda Function ImageUri from the Resource Properties\n\n        Parameters\n        ----------\n        resource_properties dict\n            Dictionary representing the Properties of the Resource\n        code_property_key str\n            Property Key of the code on the Resource\n\n        Returns\n        -------\n        str\n            Representing the local imageuri\n        '
        return cast(Optional[str], resource_properties.get(code_property_key, dict()).get('ImageUri', None))

    @staticmethod
    def _extract_sam_function_imageuri(resource_properties: Dict[str, str], code_property_key: str) -> Optional[str]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Extracts the Serverless Function ImageUri from the Resource Properties\n\n        Parameters\n        ----------\n        resource_properties dict\n            Dictionary representing the Properties of the Resource\n        code_property_key str\n            Property Key of the code on the Resource\n\n        Returns\n        -------\n        str\n            Representing the local imageuri\n        '
        return resource_properties.get(code_property_key)

    @staticmethod
    def get_template(template_dict: Dict, parameter_overrides: Optional[Dict[str, str]]=None, use_sam_transform: bool=True) -> Dict:
        if False:
            i = 10
            return i + 15
        '\n        Given a SAM template dictionary, return a cleaned copy of the template where SAM plugins have been run\n        and parameter values have been substituted.\n\n        Parameters\n        ----------\n        template_dict : dict\n            unprocessed SAM template dictionary\n\n        parameter_overrides: dict\n            Optional dictionary of values for template parameters\n\n        use_sam_transform: bool\n            Whether to transform the given template with Serverless Application Model. Default is True\n\n        Returns\n        -------\n        dict\n            Processed SAM template\n        '
        template_dict = template_dict or {}
        parameters_values = SamBaseProvider._get_parameter_values(template_dict, parameter_overrides)
        if template_dict and use_sam_transform:
            template_dict = SamTranslatorWrapper(template_dict, parameter_values=parameters_values).run_plugins()
        ResourceMetadataNormalizer.normalize(template_dict)
        resolver = IntrinsicResolver(template=template_dict, symbol_resolver=IntrinsicsSymbolTable(logical_id_translator=parameters_values, template=template_dict))
        template_dict = resolver.resolve_template(ignore_errors=True)
        return template_dict

    @staticmethod
    def get_resolved_template_dict(template_dict: Stack, parameter_overrides: Optional[Dict[str, str]]=None, normalize_resource_metadata: bool=True) -> Stack:
        if False:
            while True:
                i = 10
        "\n        Given a SAM template dictionary, return a cleaned copy of the template where SAM plugins have been run\n        and parameter values have been substituted.\n        Parameters\n        ----------\n        template_dict : dict\n            unprocessed SAM template dictionary\n        parameter_overrides: dict\n            Optional dictionary of values for template parameters\n        normalize_resource_metadata: bool\n            flag to normalize resource metadata or not; For package and deploy, we don't need to normalize resource\n            metadata, which usually exists in a CDK-synthed template and is used for build and local testing\n        Returns\n        -------\n        dict\n            Processed SAM template\n            :param template_dict:\n            :param parameter_overrides:\n            :param normalize_resource_metadata:\n        "
        template_dict = template_dict or Stack()
        parameters_values = SamBaseProvider._get_parameter_values(template_dict, parameter_overrides)
        if template_dict:
            template_dict = SamTranslatorWrapper(template_dict, parameter_values=parameters_values).run_plugins()
        if normalize_resource_metadata:
            ResourceMetadataNormalizer.normalize(template_dict)
        resolver = IntrinsicResolver(template=template_dict, symbol_resolver=IntrinsicsSymbolTable(logical_id_translator=parameters_values, template=template_dict))
        template_dict = resolver.resolve_template(ignore_errors=True)
        return template_dict

    @staticmethod
    def _get_parameter_values(template_dict: Any, parameter_overrides: Optional[Dict]) -> Dict:
        if False:
            for i in range(10):
                print('nop')
        '\n        Construct a final list of values for CloudFormation template parameters based on user-supplied values,\n        default values provided in template, and sane defaults for pseudo-parameters.\n\n        Parameters\n        ----------\n        template_dict : dict\n            SAM template dictionary\n\n        parameter_overrides : dict\n            User-supplied values for CloudFormation template parameters\n\n        Returns\n        -------\n        dict\n            Values for template parameters to substitute in template with\n        '
        default_values = SamBaseProvider._get_default_parameter_values(template_dict)
        parameter_values = {}
        parameter_values.update(IntrinsicsSymbolTable.DEFAULT_PSEUDO_PARAM_VALUES)
        parameter_values.update(default_values)
        parameter_values.update(parameter_overrides or {})
        return parameter_values

    @staticmethod
    def _get_default_parameter_values(sam_template: Dict) -> Dict:
        if False:
            while True:
                i = 10
        '\n        Method to read default values for template parameters and return it\n        Example:\n        If the template contains the following parameters defined\n        Parameters:\n            Param1:\n                Type: String\n                Default: default_value1\n            Param2:\n                Type: String\n                Default: default_value2\n\n        then, this method will grab default value for Param1 and return the following result:\n        {\n            Param1: "default_value1",\n            Param2: "default_value2"\n        }\n        :param dict sam_template: SAM template\n        :return dict: Default values for parameters\n        '
        default_values: Dict = {}
        parameter_definition = sam_template.get('Parameters', None)
        if not parameter_definition or not isinstance(parameter_definition, dict):
            LOG.debug('No Parameters detected in the template')
            return default_values
        for (param_name, value) in parameter_definition.items():
            if isinstance(value, dict) and 'Default' in value:
                default_values[param_name] = value['Default']
        LOG.debug('Collected default values for parameters: %s', default_values)
        return default_values