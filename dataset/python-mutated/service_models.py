import logging
from typing import TypedDict
from localstack.services.cloudformation.deployment_utils import check_not_found_exception
LOG = logging.getLogger(__name__)
KEY_RESOURCE_STATE = '_state_'

class DependencyNotYetSatisfied(Exception):
    """Exception indicating that a resource dependency is not (yet) deployed/available."""

    def __init__(self, resource_ids, message=None):
        if False:
            return 10
        message = message or 'Unresolved dependencies: %s' % resource_ids
        super(DependencyNotYetSatisfied, self).__init__(message)
        resource_ids = resource_ids if isinstance(resource_ids, list) else [resource_ids]
        self.resource_ids = resource_ids

class ResourceJson(TypedDict):
    Type: str
    Properties: dict

class GenericBaseModel:
    """Abstract base class representing a resource model class in LocalStack.
    This class keeps references to a combination of (1) the CF resource
    properties (as defined in the template), and (2) the current deployment
    state of a resource.

    Concrete subclasses will implement convenience methods to manage resources,
    e.g., fetching the latest deployment state, getting the resource name, etc.
    """

    def __init__(self, account_id: str, region_name: str, resource_json: dict, **params):
        if False:
            print('Hello World!')
        self.account_id = account_id
        self.region_name = region_name
        self.resource_json = resource_json
        self.resource_type = resource_json['Type']
        self.properties = resource_json['Properties'] = resource_json.get('Properties') or {}
        self.state = resource_json[KEY_RESOURCE_STATE] = resource_json.get(KEY_RESOURCE_STATE) or {}

    def fetch_state(self, stack_name, resources):
        if False:
            for i in range(10):
                print('nop')
        'Fetch the latest deployment state of this resource, or return None if not currently deployed (NOTE: THIS IS NOT ALWAYS TRUE).'
        return None

    def update_resource(self, new_resource, stack_name, resources):
        if False:
            i = 10
            return i + 15
        'Update the deployment of this resource, using the updated properties (implemented by subclasses).'
        raise NotImplementedError

    def is_updatable(self) -> bool:
        if False:
            while True:
                i = 10
        return type(self).update_resource != GenericBaseModel.update_resource

    @classmethod
    def cloudformation_type(cls):
        if False:
            while True:
                i = 10
        'Return the CloudFormation resource type name, e.g., "AWS::S3::Bucket" (implemented by subclasses).'
        pass

    @staticmethod
    def get_deploy_templates():
        if False:
            for i in range(10):
                print('nop')
        'Return template configurations used to create the final API requests (implemented by subclasses).'
        pass

    @staticmethod
    def add_defaults(resource, stack_name: str):
        if False:
            while True:
                i = 10
        'Set any defaults required, including auto-generating names. Must be called before deploying the resource'
        pass

    def fetch_and_update_state(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        if self.physical_resource_id is None:
            return None
        try:
            state = self.fetch_state(*args, **kwargs)
            self.update_state(state)
            return state
        except Exception as e:
            if not check_not_found_exception(e, self.resource_type, self.properties):
                LOG.warning('Unable to fetch state for resource %s: %s', self, e, exc_info=LOG.isEnabledFor(logging.DEBUG))

    def update_state(self, details):
        if False:
            i = 10
            return i + 15
        'Update the deployment state of this resource (existing attributes will be overwritten).'
        details = details or {}
        self.state.update(details)

    @property
    def physical_resource_id(self) -> str | None:
        if False:
            while True:
                i = 10
        'Return the (cached) physical resource ID.'
        return self.resource_json.get('PhysicalResourceId')

    @property
    def logical_resource_id(self) -> str:
        if False:
            print('Hello World!')
        'Return the logical resource ID.'
        return self.resource_json['LogicalResourceId']

    @property
    def props(self) -> dict:
        if False:
            print('Hello World!')
        'Return a copy of (1) the resource properties (from the template), combined with\n        (2) the current deployment state properties of the resource.'
        result = dict(self.properties)
        result.update(self.state or {})
        last_state = self.resource_json.get('_last_deployed_state', {})
        result.update(last_state)
        return result