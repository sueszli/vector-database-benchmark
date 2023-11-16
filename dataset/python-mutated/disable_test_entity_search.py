import unittest
from azure.cognitiveservices.search.entitysearch import EntitySearchClient
from msrest.authentication import CognitiveServicesCredentials
from azure_devtools.scenario_tests import ReplayableTest, AzureTestError
from devtools_testutils import mgmt_settings_fake as fake_settings

class EntitySearchTest(ReplayableTest):
    FILTER_HEADERS = ReplayableTest.FILTER_HEADERS + ['Ocp-Apim-Subscription-Key']

    def __init__(self, method_name):
        if False:
            i = 10
            return i + 15
        (self._fake_settings, self._real_settings) = self._load_settings()
        super(EntitySearchTest, self).__init__(method_name)

    @property
    def settings(self):
        if False:
            for i in range(10):
                print('nop')
        if self.is_live:
            if self._real_settings:
                return self._real_settings
            else:
                raise AzureTestError('Need a mgmt_settings_real.py file to run tests live.')
        else:
            return self._fake_settings

    def _load_settings(self):
        if False:
            while True:
                i = 10
        try:
            from devtools_testutils import mgmt_settings_real as real_settings
            return (fake_settings, real_settings)
        except ImportError:
            return (fake_settings, None)

    def test_search(self):
        if False:
            while True:
                i = 10
        raise unittest.SkipTest('Skipping test_search')
        query = 'seahawks'
        market = 'en-us'
        credentials = CognitiveServicesCredentials(self.settings.CS_SUBSCRIPTION_KEY)
        entity_search_api = EntitySearchClient(credentials)
        response = entity_search_api.entities.search(query=query, market=market)
        assert response is not None
        assert response._type is not None
        assert response.query_context is not None
        assert response.query_context.original_query == query
        assert response.entities is not None
        assert response.entities.value is not None
        assert len(response.entities.value) == 1
        assert response.entities.value[0].contractual_rules is not None