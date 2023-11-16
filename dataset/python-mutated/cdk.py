from localstack.services.cloudformation.service_models import GenericBaseModel
from localstack.utils.json import canonical_json
from localstack.utils.strings import md5

class CDKMetadata(GenericBaseModel):
    """Used by CDK for analytics/tracking purposes"""

    @staticmethod
    def cloudformation_type():
        if False:
            i = 10
            return i + 15
        return 'AWS::CDK::Metadata'

    def fetch_state(self, stack_name, resources):
        if False:
            i = 10
            return i + 15
        return self.props

    def update_resource(self, new_resource, stack_name, resources):
        if False:
            print('Hello World!')
        return True

    @staticmethod
    def get_deploy_templates():
        if False:
            while True:
                i = 10

        def _no_op(*args, **kwargs):
            if False:
                return 10
            pass

        def _handle_result(account_id: str, region_name: str, result: dict, logical_resource_id: str, resource: dict):
            if False:
                for i in range(10):
                    print('nop')
            resource['PhysicalResourceId'] = md5(canonical_json(resource['Properties']))
        return {'create': {'function': _no_op, 'result_handler': _handle_result}, 'delete': {'function': _no_op}}