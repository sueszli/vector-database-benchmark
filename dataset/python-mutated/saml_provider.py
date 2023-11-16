from security_monkey.cloudaux_watcher import CloudAuxWatcher
from security_monkey import AWS_DEFAULT_REGION
from cloudaux.aws.iam import list_saml_providers
from cloudaux.orchestration.aws.iam.saml_provider import get_saml_provider

class SAMLProvider(CloudAuxWatcher):
    index = 'samlprovider'
    i_am_singular = 'SAML Provider'
    i_am_plural = 'SAML Providers'
    honor_ephemerals = False
    ephemeral_paths = ['_version']
    override_region = 'universal'

    def get_name_from_list_output(self, item):
        if False:
            for i in range(10):
                print('nop')
        return item['Arn'].split('/')[-1]

    def _get_regions(self):
        if False:
            print('Hello World!')
        return [AWS_DEFAULT_REGION]

    def list_method(self, **kwargs):
        if False:
            while True:
                i = 10
        return list_saml_providers(**kwargs)

    def get_method(self, item, **kwargs):
        if False:
            return 10
        return get_saml_provider(item, **kwargs)