from ckan.lib.app_globals import app_globals as g

def test_config_not_set():
    if False:
        print('Hello World!')
    'ckan.site_about has not been configured. Behaviour has always been\n    to return an empty string.\n\n    '
    assert g.site_about == ''

def test_config_set_to_blank():
    if False:
        return 10
    'ckan.site_description is configured but with no value. Behaviour\n    has always been to return an empty string.\n\n    '
    assert g.site_description == ''