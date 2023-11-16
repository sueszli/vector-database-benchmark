import pytest
from corsheaders.middleware import ACCESS_CONTROL_ALLOW_ORIGIN
from readthedocs.search.tests.test_api import BaseTestDocumentSearch

@pytest.mark.proxito
@pytest.mark.search
class TestProxiedSearchAPI(BaseTestDocumentSearch):
    host = 'docs.readthedocs.io'

    def get_search(self, api_client, search_params):
        if False:
            print('Hello World!')
        return api_client.get(self.url, search_params, HTTP_HOST=self.host)

    def test_headers(self, api_client, project):
        if False:
            while True:
                i = 10
        version = project.versions.all().first()
        search_params = {'project': project.slug, 'version': version.slug, 'q': 'test'}
        resp = self.get_search(api_client, search_params)
        assert resp.status_code == 200
        cache_tags = f'{project.slug},{project.slug}:{version.slug},{project.slug}:rtd-search'
        assert resp['Cache-Tag'] == cache_tags
        assert ACCESS_CONTROL_ALLOW_ORIGIN not in resp.headers