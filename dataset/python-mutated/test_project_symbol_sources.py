from sentry.lang.native.sources import redact_source_secrets
from sentry.testutils.cases import APITestCase
from sentry.testutils.silo import region_silo_test
from sentry.utils import json

@region_silo_test(stable=True)
class ProjectSymbolSourcesTest(APITestCase):
    endpoint = 'sentry-api-0-project-symbol-sources'

    def test_get_successful(self):
        if False:
            i = 10
            return i + 15
        config = {'id': 'honk', 'name': 'honk source', 'layout': {'type': 'native'}, 'type': 'http', 'url': 'http://honk.beep', 'username': 'honkhonk', 'password': 'beepbeep'}
        project = self.project
        project.update_option('sentry:symbol_sources', json.dumps([config]))
        self.login_as(user=self.user)
        expected = redact_source_secrets([config])
        response = self.get_success_response(project.organization.slug, project.slug)
        assert response.data == expected
        response = self.get_success_response(project.organization.slug, project.slug, qs_params={'id': 'honk'})
        assert response.data == expected

    def test_get_unsuccessful(self):
        if False:
            while True:
                i = 10
        config = {'id': 'honk', 'name': 'honk source', 'layout': {'type': 'native'}, 'type': 'http', 'url': 'http://honk.beep', 'username': 'honkhonk', 'password': 'beepbeep'}
        project = self.project
        project.update_option('sentry:symbol_sources', json.dumps([config]))
        self.login_as(user=self.user)
        self.get_error_response(project.organization.slug, project.slug, qs_params={'id': 'hank'}, status_code=404)

@region_silo_test(stable=True)
class ProjectSymbolSourcesDeleteTest(APITestCase):
    endpoint = 'sentry-api-0-project-symbol-sources'
    method = 'delete'

    def test_delete_successful(self):
        if False:
            print('Hello World!')
        config = {'id': 'honk', 'name': 'honk source', 'layout': {'type': 'native'}, 'type': 'http', 'url': 'http://honk.beep', 'username': 'honkhonk', 'password': 'beepbeep'}
        project = self.project
        project.update_option('sentry:symbol_sources', json.dumps([config]))
        self.login_as(user=self.user)
        self.get_success_response(project.organization.slug, project.slug, qs_params={'id': 'honk'}, status=204)
        assert project.get_option('sentry:symbol_sources') == '[]'

    def test_delete_unsuccessful(self):
        if False:
            i = 10
            return i + 15
        config = {'id': 'honk', 'name': 'honk source', 'layout': {'type': 'native'}, 'type': 'http', 'url': 'http://honk.beep', 'username': 'honkhonk', 'password': 'beepbeep'}
        project = self.project
        project.update_option('sentry:symbol_sources', json.dumps([config]))
        self.login_as(user=self.user)
        self.get_error_response(project.organization.slug, project.slug, status=404)
        self.get_error_response(project.organization.slug, project.slug, qs_params={'id': 'hank'}, status=404)

@region_silo_test(stable=True)
class ProjectSymbolSourcesPostTest(APITestCase):
    endpoint = 'sentry-api-0-project-symbol-sources'
    method = 'post'

    def test_submit_successful(self):
        if False:
            for i in range(10):
                print('nop')
        config = {'id': 'honk', 'name': 'honk source', 'layout': {'type': 'native'}, 'type': 'http', 'url': 'http://honk.beep', 'username': 'honkhonk', 'password': 'beepbeep'}
        project = self.project
        self.login_as(user=self.user)
        expected = redact_source_secrets([config])[0]
        response = self.get_success_response(project.organization.slug, project.slug, raw_data=config)
        assert response.data == expected
        del config['id']
        response = self.get_success_response(project.organization.slug, project.slug, raw_data=config)
        assert 'id' in response.data

    def test_submit_duplicate(self):
        if False:
            i = 10
            return i + 15
        config = {'id': 'honk', 'name': 'honk source', 'layout': {'type': 'native'}, 'type': 'http', 'url': 'http://honk.beep', 'username': 'honkhonk', 'password': 'beepbeep'}
        project = self.project
        project.update_option('sentry:symbol_sources', json.dumps([config]))
        self.login_as(user=self.user)
        self.get_error_response(project.organization.slug, project.slug, raw_data=config)

    def test_submit_invalid_id(self):
        if False:
            for i in range(10):
                print('nop')
        config = {'id': 'sentry:project', 'name': 'honk source', 'layout': {'type': 'native'}, 'type': 'http', 'url': 'http://honk.beep', 'username': 'honkhonk', 'password': 'beepbeep'}
        project = self.project
        self.login_as(user=self.user)
        self.get_error_response(project.organization.slug, project.slug, raw_data=config)

    def test_submit_invalid_config(self):
        if False:
            while True:
                i = 10
        config = {'id': 'honk', 'name': 'honk source', 'layout': {'type': 'native'}, 'url': 'http://honk.beep', 'username': 'honkhonk', 'password': 'beepbeep'}
        project = self.project
        self.login_as(user=self.user)
        self.get_error_response(project.organization.slug, project.slug, raw_data=config)

@region_silo_test(stable=True)
class ProjectSymbolSourcesPutTest(APITestCase):
    endpoint = 'sentry-api-0-project-symbol-sources'
    method = 'put'

    def test_update_successful(self):
        if False:
            while True:
                i = 10
        config = [{'id': 'honk', 'name': 'honk source', 'layout': {'type': 'native'}, 'type': 'http', 'url': 'http://honk.beep', 'username': 'honkhonk', 'password': 'beepbeep'}, {'id': 'beep', 'name': 'beep source', 'layout': {'type': 'native'}, 'type': 'http', 'url': 'http://honk.beep', 'username': 'honkhonk', 'password': 'beepbeep'}]
        project = self.project
        project.update_option('sentry:symbol_sources', json.dumps(config))
        self.login_as(user=self.user)
        update_config = {'id': 'hank', 'name': 'honk source', 'layout': {'type': 'native'}, 'type': 'http', 'url': 'http://honk.beep', 'username': 'honkhonk', 'password': 'beepboop'}
        response = self.get_success_response(project.organization.slug, project.slug, qs_params={'id': 'honk'}, raw_data=update_config)
        assert response.data == redact_source_secrets([update_config])[0]
        update_config = {'name': 'beep source', 'layout': {'type': 'native'}, 'type': 'http', 'url': 'http://honk.beep', 'username': 'honkhonk', 'password': 'beepbeep'}
        response = self.get_success_response(project.organization.slug, project.slug, qs_params={'id': 'beep'}, raw_data=update_config)
        assert 'id' in response.data
        del response.data['id']
        assert response.data == redact_source_secrets([update_config])[0]
        source_ids = {src['id'] for src in json.loads(project.get_option('sentry:symbol_sources'))}
        assert 'hank' in source_ids
        assert 'beep' not in source_ids

    def test_update_unsuccessful(self):
        if False:
            i = 10
            return i + 15
        config = [{'id': 'honk', 'name': 'honk source', 'layout': {'type': 'native'}, 'type': 'http', 'url': 'http://honk.beep', 'username': 'honkhonk', 'password': 'beepbeep'}, {'id': 'beep', 'name': 'beep source', 'layout': {'type': 'native'}, 'type': 'http', 'url': 'http://honk.beep', 'username': 'honkhonk', 'password': 'beepbeep'}]
        project = self.project
        project.update_option('sentry:symbol_sources', json.dumps(config))
        self.login_as(user=self.user)
        update_config = {'id': 'hank', 'name': 'honk source', 'layout': {'type': 'native'}, 'url': 'http://honk.beep', 'username': 'honkhonk', 'password': 'beepboop'}
        self.get_error_response(project.organization.slug, project.slug, qs_params={'id': 'hank'}, raw_data=update_config, status=404)
        self.get_error_response(project.organization.slug, project.slug, raw_data=update_config, status=404)
        self.get_error_response(project.organization.slug, project.slug, qs_params={'id': 'honk'}, raw_data=update_config, status=400)