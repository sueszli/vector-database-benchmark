import pytest
import ckan.tests.helpers as helpers

@pytest.mark.ckan_config('ckan.plugins', u'example_iconfigurer')
@pytest.mark.usefixtures('with_plugins')
class TestExampleIConfigurer(object):

    def test_template_renders(self, app):
        if False:
            return 10
        "Our controller renders the extension's config template."
        response = app.get('/ckan-admin/myext_config_one')
        assert response.status_code == 200
        assert helpers.body_contains(response, 'My First Config Page')

    def test_config_page_has_custom_tabs(self, app):
        if False:
            print('Hello World!')
        '\n        The admin base template should include our custom ckan-admin tabs\n        added by extending ckan/templates/admin/base.html.\n        '
        response = app.get('/ckan-admin/myext_config_one', status=200)
        assert response.status_code == 200
        assert helpers.body_contains(response, 'Sysadmins')
        assert helpers.body_contains(response, 'Config')
        assert helpers.body_contains(response, 'Trash')
        assert helpers.body_contains(response, 'My First Custom Config Tab')
        assert helpers.body_contains(response, 'My Second Custom Config Tab')
        assert helpers.body_contains(response, '/ckan-admin/myext_config_one')
        assert helpers.body_contains(response, '/ckan-admin/myext_config_two')