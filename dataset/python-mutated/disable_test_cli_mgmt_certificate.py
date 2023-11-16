import unittest
import azure.mgmt.web
from devtools_testutils import AzureMgmtTestCase, RandomNameResourceGroupPreparer
from devtools_testutils.fake_credentials import FAKE_LOGIN_PASSWORD
AZURE_LOCATION = 'eastus'

class MgmtWebSiteTest(AzureMgmtTestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        super(MgmtWebSiteTest, self).setUp()
        self.mgmt_client = self.create_mgmt_client(azure.mgmt.web.WebSiteManagementClient)

    @unittest.skip("Operation returned an invalid status 'Not Found' when create certificate")
    @RandomNameResourceGroupPreparer(location=AZURE_LOCATION)
    def test_certificate(self, resource_group):
        if False:
            return 10
        RESOURCE_GROUP = resource_group.name
        NAME = 'myname'
        BODY = {'location': AZURE_LOCATION, 'host_names': ['ServerCert'], 'password': FAKE_LOGIN_PASSWORD}
        result = self.mgmt_client.certificates.create_or_update(resource_group_name=RESOURCE_GROUP, name=NAME, certificate_envelope=BODY)
        result = self.mgmt_client.certificates.get(resource_group_name=RESOURCE_GROUP, name=NAME)
        result = self.mgmt_client.certificates.list_by_resource_group(resource_group_name=RESOURCE_GROUP)
        result = self.mgmt_client.certificates.list()
        result = self.mgmt_client.certificate_registration_provider.list_operations()
        BODY = {'password': FAKE_LOGIN_PASSWORD}
        result = self.mgmt_client.certificates.update(resource_group_name=RESOURCE_GROUP, name=NAME, certificate_envelope=BODY)
        result = self.mgmt_client.certificates.delete(resource_group_name=RESOURCE_GROUP, name=NAME)