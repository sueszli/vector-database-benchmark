"""
This module makes it possible to instantiate a new Troposphere Template object
from an existing CloudFormation Template.

Usage:
    from troposphere.template_generator import TemplateGenerator
    import json

    with open("myCloudFormationTemplate.json") as f:
        json_template = json.load(f)

    template = TemplateGenerator(json_template)
    template.to_json()
"""
import importlib
import inspect
import os
import pkgutil
from collections.abc import Mapping, Sequence
from troposphere import AWSObject
from troposphere import GenericHelperFn
from troposphere import Parameter
from troposphere import AWSHelperFn, Export, Output, Ref, Tags, Template, autoscaling, cloudformation
from troposphere.policies import CreationPolicy, UpdatePolicy

class TemplateGenerator(Template):
    DEPRECATED_MODULES = ['troposphere.dynamodb2']
    EXCLUDE_MODULES = DEPRECATED_MODULES + ['troposphere.openstack.heat', 'troposphere.openstack.neutron', 'troposphere.openstack.nova']
    _inspect_members = set()
    _inspect_resources = {}
    _custom_members = set()
    _inspect_functions = {}

    def __init__(self, cf_template, **kwargs):
        if False:
            print('Hello World!')
        '\n        Instantiates a new Troposphere Template based on an existing\n        Cloudformation Template.\n        '
        super().__init__()
        if 'CustomMembers' in kwargs:
            self._custom_members = set(kwargs['CustomMembers'])
        self._reference_map = {}
        if 'AWSTemplateFormatVersion' in cf_template:
            self.set_version(cf_template['AWSTemplateFormatVersion'])
        if 'Transform' in cf_template:
            self.set_transform(cf_template['Transform'])
        if 'Description' in cf_template:
            self.set_description(cf_template['Description'])
        if 'Metadata' in cf_template:
            self.set_metadata(cf_template['Metadata'])
        for (k, v) in cf_template.get('Parameters', {}).items():
            self.add_parameter(self._create_instance(Parameter, v, k))
        for (k, v) in cf_template.get('Mappings', {}).items():
            self.add_mapping(k, self._convert_definition(v))
        for (k, v) in cf_template.get('Conditions', {}).items():
            self.add_condition(k, self._convert_definition(v, k))
        for (k, v) in cf_template.get('Resources', {}).items():
            self.add_resource(self._convert_definition(v, k, self._get_resource_type_cls(k, v)))
        for (k, v) in cf_template.get('Outputs', {}).items():
            self.add_output(self._create_instance(Output, v, k))

    @property
    def inspect_members(self):
        if False:
            i = 10
            return i + 15
        '\n        Returns the list of all troposphere members we are able to\n        construct\n        '
        if not self._inspect_members:
            TemplateGenerator._inspect_members = self._import_all_troposphere_modules()
        return self._inspect_members

    @property
    def inspect_resources(self):
        if False:
            for i in range(10):
                print('nop')
        'Returns a map of `ResourceType: ResourceClass`'
        if not self._inspect_resources:
            d = {}
            for m in self.inspect_members:
                if issubclass(m, (AWSObject, cloudformation.AWSCustomObject)) and hasattr(m, 'resource_type'):
                    d[m.resource_type] = m
            TemplateGenerator._inspect_resources = d
        return self._inspect_resources

    @property
    def inspect_functions(self):
        if False:
            return 10
        'Returns a map of `FunctionName: FunctionClass`'
        if not self._inspect_functions:
            d = {}
            for m in self.inspect_members:
                if issubclass(m, AWSHelperFn):
                    d[m.__name__] = m
            TemplateGenerator._inspect_functions = d
        return self._inspect_functions

    def _get_resource_type_cls(self, name, resource):
        if False:
            print('Hello World!')
        "Attempts to return troposphere class that represents Type of\n        provided resource. Attempts to find the troposphere class who's\n        `resource_type` field is the same as the provided resources `Type`\n        field.\n\n        :param resource: Resource to find troposphere class for\n        :return: None: If no class found for provided resource\n                 type: Type of provided resource\n        :raise ResourceTypeNotDefined:\n                  Provided resource does not have a `Type` field\n        "
        if 'Type' not in resource:
            raise ResourceTypeNotDefined(name)
        try:
            return self.inspect_resources[resource['Type']]
        except KeyError:
            for custom_member in self._custom_members:
                if custom_member.resource_type == resource['Type']:
                    return custom_member
            return None

    def _convert_definition(self, definition, ref=None, cls=None):
        if False:
            i = 10
            return i + 15
        '\n        Converts any object to its troposphere equivalent, if applicable.\n        This function will recurse into lists and mappings to create\n        additional objects as necessary.\n\n        :param {*} definition: Object to convert\n        :param str ref: Name of key in parent dict that the provided definition\n                        is from, can be None\n        :param type cls: Troposphere class which represents provided definition\n        '
        if isinstance(definition, Mapping):
            if 'Type' in definition:
                expected_type = None
                if cls is not None:
                    expected_type = cls
                else:
                    try:
                        expected_type = self._generate_custom_type(definition['Type'])
                    except TypeError:
                        if ref is not None:
                            raise ResourceTypeNotFound(ref, definition['Type'])
                        else:
                            assert not expected_type
                if expected_type:
                    args = self._normalize_properties(definition)
                    return self._create_instance(expected_type, args, ref)
            if len(definition) == 1:
                function_type = self._get_function_type(list(definition.keys())[0])
                if function_type:
                    return self._create_instance(function_type, list(definition.values())[0])
            d = {}
            for (k, v) in definition.items():
                d[k] = self._convert_definition(v)
            return d
        elif isinstance(definition, Sequence) and (not isinstance(definition, str)):
            return [self._convert_definition(v) for v in definition]
        return definition

    def _create_instance(self, cls, args, ref=None):
        if False:
            i = 10
            return i + 15
        "\n        Returns an instance of `cls` with `args` passed as arguments.\n\n        Recursively inspects `args` to create nested objects and functions as\n        necessary.\n\n        `cls` will only be considered only if it's an object we track\n         (i.e.: troposphere objects).\n\n        If `cls` has a `props` attribute, nested properties will be\n         instanciated as troposphere Property objects as necessary.\n\n        If `cls` is a list and contains a single troposphere type, the\n         returned value will be a list of instances of that type.\n        "
        if isinstance(cls, Sequence):
            if len(cls) == 1:
                if isinstance(args, str) or not isinstance(args, Sequence):
                    args = [args]
                return [self._create_instance(cls[0], v) for v in args]
        if isinstance(cls, Sequence) or cls not in self.inspect_members.union(self._custom_members):
            return self._convert_definition(args)
        elif issubclass(cls, AWSHelperFn):
            try:
                if issubclass(cls, Tags):
                    arg_dict = {}
                    for d in args:
                        arg_dict[d['Key']] = d['Value']
                    return cls(arg_dict)
                if isinstance(args, Sequence) and (not isinstance(args, str)):
                    return cls(*self._convert_definition(args))
                if issubclass(cls, autoscaling.Metadata):
                    return self._generate_autoscaling_metadata(cls, args)
                if issubclass(cls, Export):
                    return cls(args['Name'])
                args = self._convert_definition(args)
                if isinstance(args, Ref) and issubclass(cls, Ref):
                    return args
                return cls(args)
            except TypeError as ex:
                if '__init__() takes exactly' not in ex.message:
                    raise
                return GenericHelperFn(args)
        elif isinstance(args, Mapping):
            kwargs = {}
            kwargs.update(args)
            for prop_name in getattr(cls, 'props', []):
                if prop_name not in kwargs:
                    continue
                expected_type = cls.props[prop_name][0]
                if isinstance(expected_type, Sequence) or expected_type in self.inspect_members:
                    kwargs[prop_name] = self._create_instance(expected_type, kwargs[prop_name], prop_name)
                elif expected_type == bool:
                    if kwargs[prop_name] in ('True', 'true', '1'):
                        kwargs[prop_name] = True
                    elif kwargs[prop_name] in ('False', 'false', '0'):
                        kwargs[prop_name] = False
                    else:
                        kwargs[prop_name] = self._convert_definition(kwargs[prop_name], prop_name)
                else:
                    kwargs[prop_name] = self._convert_definition(kwargs[prop_name], prop_name)
            args = self._convert_definition(kwargs)
            if isinstance(args, Ref):
                return args
            if isinstance(args, AWSHelperFn):
                return self._convert_definition(kwargs)
            assert isinstance(args, Mapping)
            return cls(title=ref, **args)
        return cls(self._convert_definition(args))

    def _normalize_properties(self, definition):
        if False:
            print('Hello World!')
        '\n        Inspects the definition and returns a copy of it that is updated\n        with any special property such as Condition, UpdatePolicy and the\n        like.\n        '
        args = definition.get('Properties', {}).copy()
        if 'Condition' in definition:
            args.update({'Condition': definition['Condition']})
        if 'UpdatePolicy' in definition:
            args.update({'UpdatePolicy': self._create_instance(UpdatePolicy, definition['UpdatePolicy'])})
        if 'CreationPolicy' in definition:
            args.update({'CreationPolicy': self._create_instance(CreationPolicy, definition['CreationPolicy'])})
        if 'DeletionPolicy' in definition:
            args.update({'DeletionPolicy': self._convert_definition(definition['DeletionPolicy'])})
        if 'Metadata' in definition:
            args.update({'Metadata': self._convert_definition(definition['Metadata'])})
        if 'DependsOn' in definition:
            args.update({'DependsOn': self._convert_definition(definition['DependsOn'])})
        return args

    def _generate_custom_type(self, resource_type):
        if False:
            while True:
                i = 10
        '\n        Dynamically allocates a new CustomResource class definition using the\n        specified Custom::SomeCustomName resource type. This special resource\n        type is equivalent to the AWS::CloudFormation::CustomResource.\n        '
        if not resource_type.startswith('Custom::'):
            raise TypeError('Custom types must start with Custom::')
        custom_type = type(str(resource_type.replace('::', '')), (self.inspect_resources['AWS::CloudFormation::CustomResource'],), {'resource_type': resource_type})
        self.inspect_members.add(custom_type)
        self.inspect_resources[resource_type] = custom_type
        return custom_type

    def _generate_autoscaling_metadata(self, cls, args):
        if False:
            while True:
                i = 10
        'Provides special handling for the autoscaling.Metadata object'
        assert isinstance(args, Mapping)
        init_config = self._create_instance(cloudformation.InitConfig, args['AWS::CloudFormation::Init']['config'])
        init = self._create_instance(cloudformation.Init, {'config': init_config})
        auth = None
        if 'AWS::CloudFormation::Authentication' in args:
            auth_blocks = {}
            for k in args['AWS::CloudFormation::Authentication']:
                auth_blocks[k] = self._create_instance(cloudformation.AuthenticationBlock, args['AWS::CloudFormation::Authentication'][k], k)
            auth = self._create_instance(cloudformation.Authentication, auth_blocks)
        return cls(init, auth)

    def _get_function_type(self, function_name):
        if False:
            while True:
                i = 10
        '\n        Returns the function object that matches the provided name.\n        Only Fn:: and Ref functions are supported here so that other\n        functions specific to troposphere are skipped.\n        '
        if function_name.startswith('Fn::') and function_name[4:] in self.inspect_functions:
            return self.inspect_functions[function_name[4:]]
        return self.inspect_functions['Ref'] if function_name == 'Ref' else None

    def _import_all_troposphere_modules(self):
        if False:
            while True:
                i = 10
        'Imports all troposphere modules and returns them'
        dirname = os.path.join(os.path.dirname(__file__))
        module_names = [pkg_name for (importer, pkg_name, is_pkg) in pkgutil.walk_packages([dirname], prefix='troposphere.') if not is_pkg and pkg_name not in self.EXCLUDE_MODULES]
        module_names.append('troposphere')
        modules = []
        for name in module_names:
            modules.append(importlib.import_module(name))

        def members_predicate(m):
            if False:
                print('Hello World!')
            return inspect.isclass(m) and (not inspect.isbuiltin(m))
        members = []
        for module in modules:
            members.extend((m[1] for m in inspect.getmembers(module, members_predicate)))
        return set(members)

class ResourceTypeNotFound(Exception):

    def __init__(self, resource, resource_type):
        if False:
            return 10
        Exception.__init__(self, 'ResourceType not found for ' + resource_type + ' - ' + resource)
        self.resource_type = resource_type
        self.resource = resource

class ResourceTypeNotDefined(Exception):

    def __init__(self, resource):
        if False:
            return 10
        Exception.__init__(self, 'ResourceType not defined for ' + resource)
        self.resource = resource