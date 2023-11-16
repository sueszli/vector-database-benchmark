"""
Class that provides layers from a given SAM template
"""
import logging
from typing import Dict, List, Optional
from samcli.lib.utils.resources import AWS_LAMBDA_LAYERVERSION, AWS_SERVERLESS_LAYERVERSION
from .provider import LayerVersion, Stack
from .sam_base_provider import SamBaseProvider
from .sam_stack_provider import SamLocalStackProvider
LOG = logging.getLogger(__name__)

class SamLayerProvider(SamBaseProvider):
    """
    Fetches and returns Layers from a SAM Template. The SAM template passed to this provider is assumed to be valid,
    normalized and a dictionary.

    It may or may not contain a layer.
    """

    def __init__(self, stacks: List[Stack], use_raw_codeuri: bool=False) -> None:
        if False:
            while True:
                i = 10
        '\n        Initialize the class with SAM template data. The SAM template passed to this provider is assumed\n        to be valid, normalized and a dictionary. It should be normalized by running all pre-processing\n        before passing to this class. The process of normalization will remove structures like ``Globals``, resolve\n        intrinsic functions etc.\n        This class does not perform any syntactic validation of the template.\n\n        After the class is initialized, any changes to the ``template_dict`` will not be reflected in here.\n        You need to explicitly update the class with new template, if necessary.\n\n        Parameters\n        ----------\n        :param dict stacks: List of stacks layers are extracted from\n        :param bool use_raw_codeuri: Do not resolve adjust core_uri based on the template path, use the raw uri.\n            Note(xinhol): use_raw_codeuri is temporary to fix a bug, and will be removed for a permanent solution.\n        '
        self._stacks = stacks
        self._use_raw_codeuri = use_raw_codeuri
        self._layers = self._extract_layers()

    def get(self, name: str) -> Optional[LayerVersion]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns the layer with given name or logical id.\n        If it is in a nested stack, name can be prefixed with stack path to avoid ambiguity\n\n        Parameters\n        ----------\n        name: name or logical id of the layer.\n\n        Returns\n        -------\n        LayerVersion object of one layer.\n\n        '
        if not name:
            raise ValueError('Layer name is required')
        for layer in self._layers:
            if name in (layer.full_path, layer.layer_id, layer.name):
                return layer
        return None

    def get_all(self) -> List[LayerVersion]:
        if False:
            return 10
        '\n        Returns all Layers in template\n        Returns\n        -------\n        [LayerVersion] list of layer version objects.\n        '
        return self._layers

    def _extract_layers(self) -> List[LayerVersion]:
        if False:
            print('Hello World!')
        '\n        Extracts all resources with Type AWS::Lambda::LayerVersion and AWS::Serverless::LayerVersion and return a list\n        of those resources.\n        '
        layers = []
        for stack in self._stacks:
            for (name, resource) in stack.resources.items():
                resource_type = resource.get('Type')
                resource_properties = resource.get('Properties', {})
                if resource_type in [AWS_LAMBDA_LAYERVERSION, AWS_SERVERLESS_LAYERVERSION]:
                    code_property_key = SamBaseProvider.CODE_PROPERTY_KEYS[resource_type]
                    if SamBaseProvider._is_s3_location(resource_properties.get(code_property_key)):
                        SamBaseProvider._warn_code_extraction(resource_type, name, code_property_key)
                        continue
                    codeuri = SamBaseProvider._extract_codeuri(resource_properties, code_property_key)
                    compatible_runtimes = resource_properties.get('CompatibleRuntimes')
                    compatible_architectures = resource_properties.get('CompatibleArchitectures', None)
                    metadata = resource.get('Metadata', None)
                    layers.append(self._convert_lambda_layer_resource(stack, name, codeuri, compatible_runtimes, metadata, compatible_architectures))
        return layers

    def _convert_lambda_layer_resource(self, stack: Stack, layer_logical_id: str, codeuri: str, compatible_runtimes: Optional[List[str]], metadata: Optional[Dict], compatible_architectures: Optional[List[str]]) -> LayerVersion:
        if False:
            while True:
                i = 10
        '\n        Convert layer resource into {LayerVersion} object.\n        Parameters\n        ----------\n        stack\n        layer_logical_id\n            LogicalID of resource.\n        codeuri\n            codeuri of the layer\n        compatible_runtimes\n            list of compatible runtimes\n        metadata\n            dictionary of layer metadata\n        compatible_architectures\n            list of compatible architecture\n        Returns\n        -------\n        LayerVersion\n            The layer object\n        '
        if codeuri and (not self._use_raw_codeuri):
            LOG.debug('--base-dir is not presented, adjusting uri %s relative to %s', codeuri, stack.location)
            codeuri = SamLocalStackProvider.normalize_resource_path(stack.location, codeuri)
        return LayerVersion(layer_logical_id, codeuri, compatible_runtimes, metadata, compatible_architectures=compatible_architectures, stack_path=stack.stack_path)