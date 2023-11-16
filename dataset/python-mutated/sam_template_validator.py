"""
Library for Validating Sam Templates
"""
import functools
import logging
from typing import Dict, Optional, cast
from boto3.session import Session
from samtranslator.parser import parser
from samtranslator.public.exceptions import InvalidDocumentException
from samtranslator.translator.managed_policy_translator import ManagedPolicyLoader
from samtranslator.translator.translator import Translator
from samcli.commands.validate.lib.exceptions import InvalidSamDocumentException
from samcli.lib.utils.packagetype import IMAGE, ZIP
from samcli.lib.utils.resources import AWS_SERVERLESS_FUNCTION
from samcli.yamlhelper import yaml_dump
LOG = logging.getLogger(__name__)

class SamTemplateValidator:

    def __init__(self, sam_template: dict, managed_policy_loader: ManagedPolicyLoader, profile: Optional[str]=None, region: Optional[str]=None, parameter_overrides: Optional[dict]=None):
        if False:
            return 10
        "\n        Construct a SamTemplateValidator\n\n        Design Details:\n\n        managed_policy_loader is injected into the `__init__` to allow future expansion\n        and overriding capabilities. A typically pattern is to pass the name of the class into\n        the `__init__` as keyword args. As long as the class 'conforms' to the same 'interface'.\n        This allows the class to be changed by the client and allowing customization of the class being\n        initialized. Something I had in mind would be allowing a template to be run and checked\n        'offline' (not needing aws creds). To make this an easier transition in the future, we ingest\n        the ManagedPolicyLoader class.\n\n        Parameters\n        ----------\n        sam_template: dict\n            Dictionary representing a SAM Template\n        managed_policy_loader: ManagedPolicyLoader\n            Sam ManagedPolicyLoader\n        profile: Optional[str]\n            Optional name of boto profile\n        region: Optional[str]\n            Optional AWS region name\n        parameter_overrides: Optional[dict]\n            Template parameter overrides\n        "
        self.sam_template = sam_template
        self.managed_policy_loader = managed_policy_loader
        self.sam_parser = parser.Parser()
        self.boto3_session = Session(profile_name=profile, region_name=region)
        self.parameter_overrides = parameter_overrides or {}

    def get_translated_template_if_valid(self):
        if False:
            return 10
        '\n        Runs the SAM Translator to determine if the template provided is valid. This is similar to running a\n        ChangeSet in CloudFormation for a SAM Template\n\n        Raises\n        -------\n        InvalidSamDocumentException\n             If the template is not valid, an InvalidSamDocumentException is raised\n        '
        sam_translator = Translator(managed_policy_map=None, sam_parser=self.sam_parser, plugins=[], boto_session=self.boto3_session)
        self._replace_local_codeuri()
        self._replace_local_image()
        try:
            template = sam_translator.translate(sam_template=self.sam_template, parameter_values=self.parameter_overrides, get_managed_policy_map=self._get_managed_policy_map)
            LOG.debug('Translated template is:\n%s', yaml_dump(template))
            return yaml_dump(template)
        except InvalidDocumentException as e:
            raise InvalidSamDocumentException(functools.reduce(lambda message, error: message + ' ' + str(error), e.causes, str(e))) from e

    @functools.lru_cache(maxsize=None)
    def _get_managed_policy_map(self) -> Dict[str, str]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Helper function for getting managed policies and caching them.\n        Used by the transform for loading policies.\n\n        Returns\n        -------\n        Dict[str, str]\n            Dictionary containing the policy map\n        '
        return cast(Dict[str, str], self.managed_policy_loader.load())

    def _replace_local_codeuri(self):
        if False:
            while True:
                i = 10
        '\n        Replaces the CodeUri in AWS::Serverless::Function and DefinitionUri in AWS::Serverless::Api and\n        AWS::Serverless::HttpApi to a fake S3 Uri. This is to support running the SAM Translator with\n        valid values for these fields. If this in not done, the template is invalid in the eyes of SAM\n        Translator (the translator does not support local paths)\n        '
        all_resources = self.sam_template.get('Resources', {})
        global_settings = self.sam_template.get('Globals', {})
        for (resource_type, properties) in global_settings.items():
            if resource_type == 'Function':
                if all([_properties.get('Properties', {}).get('PackageType', ZIP) == ZIP for (_, _properties) in all_resources.items()] + [_properties.get('PackageType', ZIP) == ZIP for (_, _properties) in global_settings.items()]):
                    SamTemplateValidator._update_to_s3_uri('CodeUri', properties)
        for (_, resource) in all_resources.items():
            resource_type = resource.get('Type')
            resource_dict = resource.get('Properties', {})
            if resource_type == 'AWS::Serverless::Function' and resource_dict.get('PackageType', ZIP) == ZIP:
                SamTemplateValidator._update_to_s3_uri('CodeUri', resource_dict)
            if resource_type == 'AWS::Serverless::LayerVersion':
                SamTemplateValidator._update_to_s3_uri('ContentUri', resource_dict)
            if resource_type == 'AWS::Serverless::Api':
                if 'DefinitionUri' in resource_dict:
                    SamTemplateValidator._update_to_s3_uri('DefinitionUri', resource_dict)
            if resource_type == 'AWS::Serverless::HttpApi':
                if 'DefinitionUri' in resource_dict:
                    SamTemplateValidator._update_to_s3_uri('DefinitionUri', resource_dict)
            if resource_type == 'AWS::Serverless::StateMachine':
                if 'DefinitionUri' in resource_dict:
                    SamTemplateValidator._update_to_s3_uri('DefinitionUri', resource_dict)

    def _replace_local_image(self):
        if False:
            i = 10
            return i + 15
        '\n        Adds fake ImageUri to AWS::Serverless::Functions that reference a local image using Metadata.\n        This ensures sam validate works without having to package the app or use ImageUri.\n        '
        resources = self.sam_template.get('Resources', {})
        for (_, resource) in resources.items():
            resource_type = resource.get('Type')
            properties = resource.get('Properties', {})
            is_image_function = resource_type == AWS_SERVERLESS_FUNCTION and properties.get('PackageType') == IMAGE
            is_local_image = resource.get('Metadata', {}).get('Dockerfile')
            if is_image_function and is_local_image:
                if 'ImageUri' not in properties:
                    properties['ImageUri'] = '111111111111.dkr.ecr.region.amazonaws.com/repository'

    @staticmethod
    def is_s3_uri(uri):
        if False:
            return 10
        '\n        Checks the uri and determines if it is a valid S3 Uri\n\n        Parameters\n        ----------\n        uri str, required\n            Uri to check\n\n        Returns\n        -------\n        bool\n            Returns True if the uri given is an S3 uri, otherwise False\n\n        '
        return isinstance(uri, str) and uri.startswith('s3://')

    @staticmethod
    def _update_to_s3_uri(property_key, resource_property_dict, s3_uri_value='s3://bucket/value'):
        if False:
            i = 10
            return i + 15
        "\n        Updates the 'property_key' in the 'resource_property_dict' to the value of 's3_uri_value'\n\n        Note: The function will mutate the resource_property_dict that is pass in\n\n        Parameters\n        ----------\n        property_key str, required\n            Key in the resource_property_dict\n        resource_property_dict dict, required\n            Property dictionary of a Resource in the template to replace\n        s3_uri_value str, optional\n            Value to update the value of the property_key to\n        "
        uri_property = resource_property_dict.get(property_key, '.')
        if isinstance(uri_property, dict) or SamTemplateValidator.is_s3_uri(uri_property):
            return
        resource_property_dict[property_key] = s3_uri_value