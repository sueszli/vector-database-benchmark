import azure.mgmt.advisor
import datetime
import re
import unittest
from azure.mgmt.advisor.models import ConfigData
from devtools_testutils import AzureMgmtRecordedTestCase, ResourceGroupPreparer, recorded_by_proxy

class TestMgmtAdvisor(AzureMgmtRecordedTestCase):

    def setup_method(self, method):
        if False:
            while True:
                i = 10
        self.client = self.create_mgmt_client(azure.mgmt.advisor.AdvisorManagementClient)

    @recorded_by_proxy
    def test_generate_recommendations(self):
        if False:
            i = 10
            return i + 15

        def call(response, *args, **kwargs):
            if False:
                i = 10
                return i + 15
            return response.http_response
        response = self.client.recommendations.generate(cls=call)
        assert 'Location' in response.headers
        location = response.headers['Location']
        operation_id = re.findall('[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}', location)
        assert operation_id != None
        assert len(operation_id), 1
        response = self.client.recommendations.get_generate_status(cls=call, operation_id=operation_id[0])
        status_code = response.status_code
        assert status_code == 202 or status_code == 204

    @unittest.skip('unavailable')
    @recorded_by_proxy
    def test_suppressions(self):
        if False:
            for i in range(10):
                print('nop')
        response = list(self.client.recommendations.list())
        assert len(response) != 0
        recommendation = None
        for rec in response:
            assert rec.id != None
            assert rec.name != None
            assert rec.type != None
            assert rec.category != None
            assert rec.impact != None
            assert rec.short_description != None
            assert rec.short_description.problem != None
            assert rec.short_description.solution != None
            if rec.impacted_value != None:
                recommendation = rec
        resourceUri = recommendation.id[:recommendation.id.find('/providers/Microsoft.Advisor/recommendations')]
        recommendationName = recommendation.name
        suppressionName = 'Python_SDK_Test'
        timeToLive = '00:01:00:00'
        output = self.client.recommendations.get(resource_uri=resourceUri, recommendation_id=recommendationName)
        assert output.id == rec.id
        assert output.name == rec.name
        suppression = self.client.suppressions.create(resource_uri=resourceUri, recommendation_id=recommendationName, name=suppressionName, ttl=timeToLive)
        assert suppression.ttl == '01:00:00'
        sup = self.client.suppressions.get(resource_uri=resourceUri, recommendation_id=recommendationName, name=suppressionName)
        assert sup.name == suppressionName
        assert sup.id == resourceUri + '/providers/Microsoft.Advisor/recommendations/' + recommendationName + '/suppressions/' + suppressionName
        self.client.suppressions.delete(resource_uri=resourceUri, recommendation_id=recommendationName, name=suppressionName)

    @unittest.skip('unavailable')
    @recorded_by_proxy
    def test_configurations_subscription(self):
        if False:
            print('Hello World!')
        input = ConfigData()
        input.low_cpu_threshold = 20
        response = self.client.configurations.create_in_subscription(input)
        output = list(self.client.configurations.list_by_subscription())[0]
        assert output.low_cpu_threshold == '20'
        input.low_cpu_threshold = 5
        response = self.client.configurations.create_in_subscription(input)
        output = list(self.client.configurations.list_by_subscription())[0]
        assert output.low_cpu_threshold == '5'

    @ResourceGroupPreparer()
    @recorded_by_proxy
    def test_configurations_resourcegroup(self, resource_group):
        if False:
            while True:
                i = 10
        resourceGroupName = resource_group.name
        configurationName = 'default'
        input = ConfigData()
        input.exclude = True
        self.client.configurations.create_in_resource_group(configuration_name=configurationName, resource_group=resourceGroupName, config_contract=input)
        output = list(self.client.configurations.list_by_resource_group(resource_group=resourceGroupName))[0]
        assert output.exclude == True
        input.exclude = False
        self.client.configurations.create_in_resource_group(configuration_name=configurationName, resource_group=resourceGroupName, config_contract=input)
        output = list(self.client.configurations.list_by_resource_group(resource_group=resourceGroupName))[0]
        assert output.exclude == False
if __name__ == '__main__':
    unittest.main()