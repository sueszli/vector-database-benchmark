"""
Class that Normalizes a Template based on Resource Metadata
"""
import json
import logging
import re
from copy import deepcopy
from pathlib import Path
from typing import Dict
from samcli.lib.iac.cdk.utils import is_cdk_project
from samcli.lib.utils.resources import AWS_CLOUDFORMATION_STACK
CDK_NESTED_STACK_RESOURCE_ID_SUFFIX = '.NestedStack'
RESOURCES_KEY = 'Resources'
PROPERTIES_KEY = 'Properties'
METADATA_KEY = 'Metadata'
RESOURCE_CDK_PATH_METADATA_KEY = 'aws:cdk:path'
ASSET_PATH_METADATA_KEY = 'aws:asset:path'
ASSET_PROPERTY_METADATA_KEY = 'aws:asset:property'
IMAGE_ASSET_PROPERTY = 'Code.ImageUri'
ASSET_DOCKERFILE_PATH_KEY = 'aws:asset:dockerfile-path'
ASSET_DOCKERFILE_BUILD_ARGS_KEY = 'aws:asset:docker-build-args'
SAM_RESOURCE_ID_KEY = 'SamResourceId'
SAM_IS_NORMALIZED = 'SamNormalized'
SAM_METADATA_DOCKERFILE_KEY = 'Dockerfile'
SAM_METADATA_DOCKER_CONTEXT_KEY = 'DockerContext'
SAM_METADATA_DOCKER_BUILD_ARGS_KEY = 'DockerBuildArgs'
ASSET_BUNDLED_METADATA_KEY = 'aws:asset:is-bundled'
SAM_METADATA_SKIP_BUILD_KEY = 'SkipBuild'
CDK_ASSET_PARAMETER_PATTERN = re.compile('^AssetParameters[0-9a-fA-F]{64}(?:S3Bucket|S3VersionKey|ArtifactHash)[0-9a-fA-F]{8}$')
BUILD_PROPERTIES_PASCAL_TO_SNAKE_CASE_PATTERN = re.compile('(?<!^)(?=[A-Z])')
LOG = logging.getLogger(__name__)

class ResourceMetadataNormalizer:

    @staticmethod
    def normalize(template_dict, normalize_parameters=False):
        if False:
            return 10
        '\n        Normalize all Resources in the template with the Metadata Key on the resource.\n\n        This method will mutate the template\n\n        Parameters\n        ----------\n        template_dict dict\n            Dictionary representing the template\n\n        '
        resources = template_dict.get(RESOURCES_KEY, {})
        for (logical_id, resource) in resources.items():
            resource_metadata = deepcopy(resource.get(METADATA_KEY)) or {}
            is_normalized = resource_metadata.get(SAM_IS_NORMALIZED, False)
            if not is_normalized:
                asset_property = resource_metadata.get(ASSET_PROPERTY_METADATA_KEY)
                if asset_property == IMAGE_ASSET_PROPERTY:
                    asset_metadata = ResourceMetadataNormalizer._extract_image_asset_metadata(resource_metadata)
                    ResourceMetadataNormalizer._update_resource_metadata(resource_metadata, asset_metadata)
                    asset_path = logical_id.lower()
                else:
                    asset_path = resource_metadata.get(ASSET_PATH_METADATA_KEY)
                ResourceMetadataNormalizer._replace_property(asset_property, asset_path, resource, logical_id)
                if asset_path and asset_property:
                    resource_metadata[SAM_IS_NORMALIZED] = True
            skip_build = resource_metadata.get(ASSET_BUNDLED_METADATA_KEY, False)
            if skip_build:
                ResourceMetadataNormalizer._update_resource_metadata(resource_metadata, {SAM_METADATA_SKIP_BUILD_KEY: True})
            ResourceMetadataNormalizer._update_resource_metadata(resource_metadata, {SAM_RESOURCE_ID_KEY: ResourceMetadataNormalizer.get_resource_id(resource, logical_id)})
            resource[METADATA_KEY] = resource_metadata
        if normalize_parameters and is_cdk_project(template_dict):
            resources_copy = {logical_id: resource for (logical_id, resource) in resources.items() if resource.get('Type', '') != AWS_CLOUDFORMATION_STACK}
            resources_as_string = json.dumps(resources_copy)
            parameters = template_dict.get('Parameters', {})
            default_value = ' '
            for (parameter_name, parameter_value) in parameters.items():
                parameter_name_match = CDK_ASSET_PARAMETER_PATTERN.match(parameter_name)
                if parameter_name_match and 'Default' not in parameter_value and (parameter_value.get('Type', '') == 'String') and (f'"Ref": "{parameter_name}"' not in resources_as_string):
                    LOG.debug("set default value for parameter %s to '%s'", parameter_name, default_value)
                    parameter_value['Default'] = default_value

    @staticmethod
    def _replace_property(property_key, property_value, resource, logical_id):
        if False:
            return 10
        '\n        Replace a property with an asset on a given resource\n\n        This method will mutate the template\n\n        Parameters\n        ----------\n        property str\n            The property to replace on the resource\n        property_value str\n            The new value of the property\n        resource dict\n            Dictionary representing the Resource to change\n        logical_id str\n            LogicalId of the Resource\n\n        '
        if property_key and property_value:
            nested_keys = property_key.split('.')
            target_dict = resource.get(PROPERTIES_KEY, {})
            while len(nested_keys) > 1:
                key = nested_keys.pop(0)
                target_dict[key] = {}
                target_dict = target_dict[key]
            target_dict[nested_keys[0]] = property_value
        elif property_key or property_value:
            LOG.info('WARNING: Ignoring Metadata for Resource %s. Metadata contains only aws:asset:path or aws:assert:property but not both', logical_id)

    @staticmethod
    def _extract_image_asset_metadata(metadata):
        if False:
            while True:
                i = 10
        '\n        Extract/create relevant metadata properties for image assets\n\n        Parameters\n        ----------\n        metadata dict\n            Metadata to use for extracting image assets properties\n\n        Returns\n        -------\n        dict\n            metadata properties for image-type lambda function\n\n        '
        asset_path = Path(metadata.get(ASSET_PATH_METADATA_KEY, ''))
        dockerfile_path = Path(metadata.get(ASSET_DOCKERFILE_PATH_KEY), '')
        return {SAM_METADATA_DOCKERFILE_KEY: str(dockerfile_path.as_posix()), SAM_METADATA_DOCKER_CONTEXT_KEY: str(asset_path), SAM_METADATA_DOCKER_BUILD_ARGS_KEY: metadata.get(ASSET_DOCKERFILE_BUILD_ARGS_KEY, {})}

    @staticmethod
    def _update_resource_metadata(metadata, updated_values):
        if False:
            print('Hello World!')
        '\n        Update the metadata values for image-type lambda functions\n\n        This method will mutate the template\n\n        Parameters\n        ----------\n        metadata dict\n            Metadata dict to be updated\n        updated_values dict\n            Dict of key-value pairs to append to the existing metadata\n\n        '
        for (key, val) in updated_values.items():
            metadata[key] = val

    @staticmethod
    def get_resource_id(resource_properties, logical_id):
        if False:
            return 10
        '\n        Get unique id for a resource.\n        for any resource, the resource id can be the customer defined id if exist, if not exist it can be the\n        cdk-defined resource id, or the logical id if the resource id is not found.\n\n        Parameters\n        ----------\n        resource_properties dict\n            Properties of this resource\n        logical_id str\n            LogicalID of the resource\n\n        Returns\n        -------\n        str\n            The unique function id\n        '
        resource_metadata = resource_properties.get('Metadata', {})
        customer_defined_id = resource_metadata.get(SAM_RESOURCE_ID_KEY)
        if isinstance(customer_defined_id, str) and customer_defined_id:
            LOG.debug('Sam customer defined id is more priority than other IDs. Customer defined id for resource %s is %s', logical_id, customer_defined_id)
            return customer_defined_id
        resource_cdk_path = resource_metadata.get(RESOURCE_CDK_PATH_METADATA_KEY)
        if not isinstance(resource_cdk_path, str) or not resource_cdk_path:
            LOG.debug('There is no customer defined id or cdk path defined for resource %s, so we will use the resource logical id as the resource id', logical_id)
            return logical_id
        cdk_path_partitions = resource_cdk_path.split('/')
        min_cdk_path_partitions_length = 2
        LOG.debug('CDK Path for resource %s is %s', logical_id, cdk_path_partitions)
        if len(cdk_path_partitions) < min_cdk_path_partitions_length:
            LOG.warning("Cannot detect function id from aws:cdk:path metadata '%s', using default logical id", resource_cdk_path)
            return logical_id
        cdk_resource_id = cdk_path_partitions[-2] if cdk_path_partitions[-1] == 'Resource' or (resource_properties.get('Type', '') == AWS_CLOUDFORMATION_STACK and cdk_path_partitions[-2].endswith(CDK_NESTED_STACK_RESOURCE_ID_SUFFIX)) else cdk_path_partitions[-1]
        if resource_properties.get('Type', '') == AWS_CLOUDFORMATION_STACK and cdk_resource_id.endswith(CDK_NESTED_STACK_RESOURCE_ID_SUFFIX):
            cdk_resource_id = cdk_resource_id[:-len(CDK_NESTED_STACK_RESOURCE_ID_SUFFIX)]
        return cdk_resource_id

    @staticmethod
    def normalize_build_properties(build_props) -> Dict:
        if False:
            i = 10
            return i + 15
        '\n        Convert PascalCase properties in the template to snake case to be consistent with\n        what Lambda Builders expects from its properties\n\n        :param build_props: Properties to be passed to Lambda Builders\n        :return: dict of normalized properties\n        '
        normalized_props = {}
        for (key, val) in build_props.items():
            normalized_key = BUILD_PROPERTIES_PASCAL_TO_SNAKE_CASE_PATTERN.sub('_', key).lower()
            normalized_props[normalized_key] = val
        return normalized_props