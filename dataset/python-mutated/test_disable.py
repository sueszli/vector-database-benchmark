import pytest
import ckan.plugins as p

@pytest.mark.ckan_config('ckan.datastore.sqlsearch.enabled', None)
def test_disabled_by_default():
    if False:
        while True:
            i = 10
    with p.use_plugin('datastore'):
        with pytest.raises(KeyError, match=u"Action 'datastore_search_sql' not found"):
            p.toolkit.get_action('datastore_search_sql')

@pytest.mark.ckan_config('ckan.datastore.sqlsearch.enabled', False)
def test_disable_sql_search():
    if False:
        return 10
    with p.use_plugin('datastore'):
        with pytest.raises(KeyError, match=u"Action 'datastore_search_sql' not found"):
            p.toolkit.get_action('datastore_search_sql')

@pytest.mark.ckan_config('ckan.datastore.sqlsearch.enabled', True)
def test_enabled_sql_search():
    if False:
        i = 10
        return i + 15
    with p.use_plugin('datastore'):
        p.toolkit.get_action('datastore_search_sql')