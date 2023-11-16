import jmespath
from botocore import xform_name
from .params import get_data_member

def all_not_none(iterable):
    if False:
        while True:
            i = 10
    '\n    Return True if all elements of the iterable are not None (or if the\n    iterable is empty). This is like the built-in ``all``, except checks\n    against None, so 0 and False are allowable values.\n    '
    for element in iterable:
        if element is None:
            return False
    return True

def build_identifiers(identifiers, parent, params=None, raw_response=None):
    if False:
        print('Hello World!')
    '\n    Builds a mapping of identifier names to values based on the\n    identifier source location, type, and target. Identifier\n    values may be scalars or lists depending on the source type\n    and location.\n\n    :type identifiers: list\n    :param identifiers: List of :py:class:`~boto3.resources.model.Parameter`\n                        definitions\n    :type parent: ServiceResource\n    :param parent: The resource instance to which this action is attached.\n    :type params: dict\n    :param params: Request parameters sent to the service.\n    :type raw_response: dict\n    :param raw_response: Low-level operation response.\n    :rtype: list\n    :return: An ordered list of ``(name, value)`` identifier tuples.\n    '
    results = []
    for identifier in identifiers:
        source = identifier.source
        target = identifier.target
        if source == 'response':
            value = jmespath.search(identifier.path, raw_response)
        elif source == 'requestParameter':
            value = jmespath.search(identifier.path, params)
        elif source == 'identifier':
            value = getattr(parent, xform_name(identifier.name))
        elif source == 'data':
            value = get_data_member(parent, identifier.path)
        elif source == 'input':
            continue
        else:
            raise NotImplementedError(f'Unsupported source type: {source}')
        results.append((xform_name(target), value))
    return results

def build_empty_response(search_path, operation_name, service_model):
    if False:
        return 10
    "\n    Creates an appropriate empty response for the type that is expected,\n    based on the service model's shape type. For example, a value that\n    is normally a list would then return an empty list. A structure would\n    return an empty dict, and a number would return None.\n\n    :type search_path: string\n    :param search_path: JMESPath expression to search in the response\n    :type operation_name: string\n    :param operation_name: Name of the underlying service operation.\n    :type service_model: :ref:`botocore.model.ServiceModel`\n    :param service_model: The Botocore service model\n    :rtype: dict, list, or None\n    :return: An appropriate empty value\n    "
    response = None
    operation_model = service_model.operation_model(operation_name)
    shape = operation_model.output_shape
    if search_path:
        for item in search_path.split('.'):
            item = item.strip('[0123456789]$')
            if shape.type_name == 'structure':
                shape = shape.members[item]
            elif shape.type_name == 'list':
                shape = shape.member
            else:
                raise NotImplementedError('Search path hits shape type {} from {}'.format(shape.type_name, item))
    if shape.type_name == 'structure':
        response = {}
    elif shape.type_name == 'list':
        response = []
    elif shape.type_name == 'map':
        response = {}
    return response

class RawHandler:
    """
    A raw action response handler. This passed through the response
    dictionary, optionally after performing a JMESPath search if one
    has been defined for the action.

    :type search_path: string
    :param search_path: JMESPath expression to search in the response
    :rtype: dict
    :return: Service response
    """

    def __init__(self, search_path):
        if False:
            for i in range(10):
                print('nop')
        self.search_path = search_path

    def __call__(self, parent, params, response):
        if False:
            print('Hello World!')
        '\n        :type parent: ServiceResource\n        :param parent: The resource instance to which this action is attached.\n        :type params: dict\n        :param params: Request parameters sent to the service.\n        :type response: dict\n        :param response: Low-level operation response.\n        '
        if self.search_path and self.search_path != '$':
            response = jmespath.search(self.search_path, response)
        return response

class ResourceHandler:
    """
    Creates a new resource or list of new resources from the low-level
    response based on the given response resource definition.

    :type search_path: string
    :param search_path: JMESPath expression to search in the response

    :type factory: ResourceFactory
    :param factory: The factory that created the resource class to which
                    this action is attached.

    :type resource_model: :py:class:`~boto3.resources.model.ResponseResource`
    :param resource_model: Response resource model.

    :type service_context: :py:class:`~boto3.utils.ServiceContext`
    :param service_context: Context about the AWS service

    :type operation_name: string
    :param operation_name: Name of the underlying service operation, if it
                           exists.

    :rtype: ServiceResource or list
    :return: New resource instance(s).
    """

    def __init__(self, search_path, factory, resource_model, service_context, operation_name=None):
        if False:
            return 10
        self.search_path = search_path
        self.factory = factory
        self.resource_model = resource_model
        self.operation_name = operation_name
        self.service_context = service_context

    def __call__(self, parent, params, response):
        if False:
            i = 10
            return i + 15
        '\n        :type parent: ServiceResource\n        :param parent: The resource instance to which this action is attached.\n        :type params: dict\n        :param params: Request parameters sent to the service.\n        :type response: dict\n        :param response: Low-level operation response.\n        '
        resource_name = self.resource_model.type
        json_definition = self.service_context.resource_json_definitions.get(resource_name)
        resource_cls = self.factory.load_from_definition(resource_name=resource_name, single_resource_json_definition=json_definition, service_context=self.service_context)
        raw_response = response
        search_response = None
        if self.search_path:
            search_response = jmespath.search(self.search_path, raw_response)
        identifiers = dict(build_identifiers(self.resource_model.identifiers, parent, params, raw_response))
        plural = [v for v in identifiers.values() if isinstance(v, list)]
        if plural:
            response = []
            for i in range(len(plural[0])):
                response_item = None
                if search_response:
                    response_item = search_response[i]
                response.append(self.handle_response_item(resource_cls, parent, identifiers, response_item))
        elif all_not_none(identifiers.values()):
            response = self.handle_response_item(resource_cls, parent, identifiers, search_response)
        else:
            response = None
            if self.operation_name is not None:
                response = build_empty_response(self.search_path, self.operation_name, self.service_context.service_model)
        return response

    def handle_response_item(self, resource_cls, parent, identifiers, resource_data):
        if False:
            return 10
        '\n        Handles the creation of a single response item by setting\n        parameters and creating the appropriate resource instance.\n\n        :type resource_cls: ServiceResource subclass\n        :param resource_cls: The resource class to instantiate.\n        :type parent: ServiceResource\n        :param parent: The resource instance to which this action is attached.\n        :type identifiers: dict\n        :param identifiers: Map of identifier names to value or values.\n        :type resource_data: dict or None\n        :param resource_data: Data for resource attributes.\n        :rtype: ServiceResource\n        :return: New resource instance.\n        '
        kwargs = {'client': parent.meta.client}
        for (name, value) in identifiers.items():
            if isinstance(value, list):
                value = value.pop(0)
            kwargs[name] = value
        resource = resource_cls(**kwargs)
        if resource_data is not None:
            resource.meta.data = resource_data
        return resource