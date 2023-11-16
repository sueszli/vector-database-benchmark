import pytest
import ckan.lib.create_test_data as ctd
import ckan.plugins as p
import ckan.tests.factories as factories
from ckan.logic import check_access, NotAuthorized
from ckan.logic.auth.get import user_list as core_user_list
auth_message = u'No search for you'
user_list_message = u'Nothing to see here'

class AuthTestException(Exception):
    pass

@p.toolkit.chained_auth_function
def datastore_search_sql_auth(up_func, context, data_dict):
    if False:
        while True:
            i = 10
    assert up_func.auth_allow_anonymous_access
    assert up_func(context, data_dict) == {u'success': True}
    raise AuthTestException(auth_message)

@p.toolkit.chained_auth_function
def user_list(next_auth, context, data_dict):
    if False:
        return 10
    assert next_auth == core_user_list
    raise AuthTestException(user_list_message)

@p.toolkit.chained_auth_function
def user_create(next_auth, context, data_dict):
    if False:
        while True:
            i = 10
    return next_auth(context, data_dict)

class ExampleDataStoreSearchSQLPlugin(p.SingletonPlugin):
    p.implements(p.IAuthFunctions)

    def get_auth_functions(self):
        if False:
            i = 10
            return i + 15
        return {u'datastore_search_sql': datastore_search_sql_auth, u'user_list': user_list}

@pytest.mark.ckan_config(u'ckan.plugins', u'datastore example_data_store_search_sql_plugin')
@pytest.mark.usefixtures('with_plugins', 'clean_db')
class TestChainedAuth(object):

    def test_datastore_search_sql_auth(self):
        if False:
            for i in range(10):
                print('nop')
        ctd.CreateTestData.create()
        with pytest.raises(AuthTestException) as raise_context:
            check_access(u'datastore_search_sql', {u'user': u'annafan', u'table_names': []}, {})
        assert raise_context.value.args == (auth_message,)

    def test_chain_core_auth_functions(self):
        if False:
            for i in range(10):
                print('nop')
        user = factories.User()
        context = {u'user': user[u'name']}
        with pytest.raises(AuthTestException) as raise_context:
            check_access(u'user_list', context, {})
        assert raise_context.value.args == (user_list_message,)
        with pytest.raises(NotAuthorized):
            check_access(u'user_list', {u'ignore_auth': False, u'user': u'not_a_real_user'}, {})

class ExampleExternalProviderPlugin(p.SingletonPlugin):
    p.implements(p.IAuthFunctions)

    def get_auth_functions(self):
        if False:
            i = 10
            return i + 15
        return {u'user_create': user_create}

@pytest.mark.ckan_config(u'ckan.plugins', u'datastore example_data_store_search_sql_plugin')
@pytest.mark.usefixtures(u'with_plugins', u'clean_db')
class TestChainedAuthBuiltInFallback(object):

    @pytest.mark.ckan_config('ckan.auth.create_user_via_web', True)
    def test_user_create_chained_auth(self):
        if False:
            while True:
                i = 10
        ctd.CreateTestData.create()
        check_access(u'user_create', {u'user': u'annafan'}, {})