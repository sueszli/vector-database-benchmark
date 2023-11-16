import pytest
import ckan.tests.factories as factories
import ckan.tests.helpers as helpers
from ckan.plugins.toolkit import ValidationError, NotAuthorized
from ckanext.datastore.tests.helpers import when_was_last_analyze

def _search(resource_id):
    if False:
        while True:
            i = 10
    return helpers.call_action(u'datastore_search', resource_id=resource_id)

class TestDatastoreUpsert(object):

    @pytest.mark.ckan_config('ckan.plugins', 'datastore')
    @pytest.mark.usefixtures('clean_datastore', 'with_plugins')
    def test_upsert_requires_auth(self):
        if False:
            print('Hello World!')
        resource = factories.Resource(url_type=u'datastore')
        data = {'resource_id': resource['id'], 'force': True, 'primary_key': 'id', 'fields': [{'id': 'id', 'type': 'text'}, {'id': 'book', 'type': 'text'}], 'records': []}
        helpers.call_action('datastore_create', **data)
        data = {'resource_id': resource['id']}
        with pytest.raises(NotAuthorized) as context:
            helpers.call_action('datastore_upsert', context={'user': '', 'ignore_auth': False}, **data)
        assert u'Action datastore_upsert requires an authenticated user' in str(context.value)

    @pytest.mark.ckan_config('ckan.plugins', 'datastore')
    @pytest.mark.usefixtures('clean_datastore', 'with_plugins')
    def test_upsert_empty_fails(self):
        if False:
            for i in range(10):
                print('nop')
        resource = factories.Resource(url_type=u'datastore')
        data = {'resource_id': resource['id'], 'force': True, 'primary_key': 'id', 'fields': [{'id': 'id', 'type': 'text'}, {'id': 'book', 'type': 'text'}], 'records': []}
        helpers.call_action('datastore_create', **data)
        data = {}
        with pytest.raises(ValidationError) as context:
            helpers.call_action('datastore_upsert', **data)
        assert u"'Missing value'" in str(context.value)

    @pytest.mark.ckan_config('ckan.plugins', 'datastore')
    @pytest.mark.usefixtures('clean_datastore', 'with_plugins')
    def test_basic_as_update(self):
        if False:
            return 10
        resource = factories.Resource()
        data = {'resource_id': resource['id'], 'force': True, 'primary_key': 'id', 'fields': [{'id': 'id', 'type': 'text'}, {'id': 'book', 'type': 'text'}, {'id': 'author', 'type': 'text'}], 'records': [{'id': '1', 'book': u'El Niño', 'author': 'Torres'}]}
        helpers.call_action('datastore_create', **data)
        data = {'resource_id': resource['id'], 'force': True, 'method': 'upsert', 'records': [{'id': '1', 'book': u'The boy', 'author': u'F Torres'}]}
        helpers.call_action('datastore_upsert', **data)
        search_result = _search(resource['id'])
        assert search_result['total'] == 1
        assert search_result['records'][0]['book'] == 'The boy'
        assert search_result['records'][0]['author'] == 'F Torres'

    @pytest.mark.ckan_config('ckan.plugins', 'datastore')
    @pytest.mark.usefixtures('clean_datastore', 'with_plugins')
    def test_basic_as_insert(self):
        if False:
            return 10
        resource = factories.Resource()
        data = {'resource_id': resource['id'], 'force': True, 'primary_key': 'id', 'fields': [{'id': 'id', 'type': 'text'}, {'id': 'book', 'type': 'text'}, {'id': 'author', 'type': 'text'}], 'records': [{'id': '1', 'book': u'El Niño', 'author': 'Torres'}]}
        helpers.call_action('datastore_create', **data)
        data = {'resource_id': resource['id'], 'force': True, 'method': 'upsert', 'records': [{'id': '2', 'book': u'The boy', 'author': u'F Torres'}]}
        helpers.call_action('datastore_upsert', **data)
        search_result = _search(resource['id'])
        assert search_result['total'] == 2
        assert search_result['records'][0]['book'] == u'El Niño'
        assert search_result['records'][1]['book'] == u'The boy'

    @pytest.mark.ckan_config('ckan.plugins', 'datastore')
    @pytest.mark.usefixtures('clean_datastore', 'with_plugins')
    def test_upsert_only_one_field(self):
        if False:
            for i in range(10):
                print('nop')
        resource = factories.Resource()
        data = {'resource_id': resource['id'], 'force': True, 'primary_key': 'id', 'fields': [{'id': 'id', 'type': 'text'}, {'id': 'book', 'type': 'text'}, {'id': 'author', 'type': 'text'}], 'records': [{'id': '1', 'book': u'El Niño', 'author': 'Torres'}]}
        helpers.call_action('datastore_create', **data)
        data = {'resource_id': resource['id'], 'force': True, 'method': 'upsert', 'records': [{'id': '1', 'book': u'The boy'}]}
        helpers.call_action('datastore_upsert', **data)
        search_result = _search(resource['id'])
        assert search_result['total'] == 1
        assert search_result['records'][0]['book'] == 'The boy'
        assert search_result['records'][0]['author'] == 'Torres'

    @pytest.mark.ckan_config('ckan.plugins', 'datastore')
    @pytest.mark.usefixtures('clean_datastore', 'with_plugins')
    def test_field_types(self):
        if False:
            print('Hello World!')
        resource = factories.Resource(url_type='datastore')
        data = {'resource_id': resource['id'], 'force': True, 'primary_key': u'bük', 'fields': [{'id': u'bük', 'type': 'text'}, {'id': 'author', 'type': 'text'}, {'id': 'nested', 'type': 'json'}, {'id': 'characters', 'type': 'text[]'}, {'id': 'published'}], 'records': [{u'bük': 'annakarenina', 'author': 'tolstoy', 'published': '2005-03-01', 'nested': ['b', {'moo': 'moo'}]}, {u'bük': 'warandpeace', 'author': 'tolstoy', 'nested': {'a': 'b'}}, {'author': 'adams', 'characters': ['Arthur', 'Marvin'], 'nested': {'foo': 'bar'}, u'bük': u'guide to the galaxy'}]}
        helpers.call_action('datastore_create', **data)
        data = {'resource_id': resource['id'], 'method': 'upsert', 'records': [{'author': 'adams', 'characters': ['Bob', 'Marvin'], 'nested': {'baz': 3}, u'bük': u'guide to the galaxy'}]}
        helpers.call_action('datastore_upsert', **data)
        search_result = _search(resource['id'])
        assert search_result['total'] == 3
        assert search_result['records'][0]['published'] == u'2005-03-01T00:00:00'
        assert search_result['records'][2]['author'] == 'adams'
        assert search_result['records'][2]['characters'] == ['Bob', 'Marvin']
        assert search_result['records'][2]['nested'] == {'baz': 3}

    @pytest.mark.ckan_config('ckan.plugins', 'datastore')
    @pytest.mark.usefixtures('clean_datastore', 'with_plugins')
    def test_percent(self):
        if False:
            for i in range(10):
                print('nop')
        resource = factories.Resource()
        data = {'resource_id': resource['id'], 'force': True, 'primary_key': 'id', 'fields': [{'id': 'id', 'type': 'text'}, {'id': 'bo%ok', 'type': 'text'}, {'id': 'author', 'type': 'text'}], 'records': [{'id': '1%', 'bo%ok': u'El Niño', 'author': 'Torres'}]}
        helpers.call_action('datastore_create', **data)
        data = {'resource_id': resource['id'], 'force': True, 'method': 'upsert', 'records': [{'id': '1%', 'bo%ok': u'The % boy', 'author': u'F Torres'}, {'id': '2%', 'bo%ok': u'Gu%ide', 'author': u'Adams'}]}
        helpers.call_action('datastore_upsert', **data)
        search_result = _search(resource['id'])
        assert search_result['total'] == 2
        assert search_result['records'][0]['bo%ok'] == 'The % boy'
        assert search_result['records'][1]['bo%ok'] == 'Gu%ide'

    @pytest.mark.ckan_config('ckan.plugins', 'datastore')
    @pytest.mark.usefixtures('clean_datastore', 'with_plugins')
    def test_missing_key(self):
        if False:
            for i in range(10):
                print('nop')
        resource = factories.Resource()
        data = {'resource_id': resource['id'], 'force': True, 'primary_key': 'id', 'fields': [{'id': 'id', 'type': 'text'}, {'id': 'book', 'type': 'text'}, {'id': 'author', 'type': 'text'}], 'records': [{'id': '1', 'book': 'guide', 'author': 'adams'}]}
        helpers.call_action('datastore_create', **data)
        data = {'resource_id': resource['id'], 'force': True, 'method': 'upsert', 'records': [{'book': u'El Niño', 'author': 'Torres'}]}
        with pytest.raises(ValidationError) as context:
            helpers.call_action('datastore_upsert', **data)
        assert u'fields "id" are missing' in str(context.value)

    @pytest.mark.ckan_config('ckan.plugins', 'datastore')
    @pytest.mark.usefixtures('clean_datastore', 'with_plugins')
    def test_non_existing_field(self):
        if False:
            while True:
                i = 10
        resource = factories.Resource(url_type='datastore')
        data = {'resource_id': resource['id'], 'force': True, 'primary_key': u'id', 'fields': [{'id': 'id', 'type': 'text'}, {'id': 'book', 'type': 'text'}, {'id': 'author', 'type': 'text'}], 'records': []}
        helpers.call_action('datastore_create', **data)
        data = {'resource_id': resource['id'], 'method': 'upsert', 'records': [{'id': '1', 'dummy': 'tolkien'}]}
        with pytest.raises(ValidationError) as context:
            helpers.call_action('datastore_upsert', **data)
        assert u'fields "dummy" do not exist' in str(context.value)

    @pytest.mark.ckan_config('ckan.plugins', 'datastore')
    @pytest.mark.usefixtures('clean_datastore', 'with_plugins')
    def test_upsert_works_with_empty_list_in_json_field(self):
        if False:
            return 10
        resource = factories.Resource()
        data = {'resource_id': resource['id'], 'force': True, 'primary_key': 'id', 'fields': [{'id': 'id', 'type': 'text'}, {'id': 'nested', 'type': 'json'}], 'records': [{'id': '1', 'nested': {'foo': 'bar'}}]}
        helpers.call_action('datastore_create', **data)
        data = {'resource_id': resource['id'], 'force': True, 'method': 'upsert', 'records': [{'id': '1', 'nested': []}]}
        helpers.call_action('datastore_upsert', **data)
        search_result = _search(resource['id'])
        assert search_result['total'] == 1
        assert search_result['records'][0]['nested'] == []

    @pytest.mark.ckan_config('ckan.plugins', 'datastore')
    @pytest.mark.usefixtures('clean_datastore', 'with_plugins')
    def test_delete_field_value(self):
        if False:
            for i in range(10):
                print('nop')
        resource = factories.Resource()
        data = {'resource_id': resource['id'], 'force': True, 'primary_key': 'id', 'fields': [{'id': 'id', 'type': 'text'}, {'id': 'book', 'type': 'text'}, {'id': 'author', 'type': 'text'}], 'records': [{'id': '1', 'book': u'El Niño', 'author': 'Torres'}]}
        helpers.call_action('datastore_create', **data)
        data = {'resource_id': resource['id'], 'force': True, 'method': 'upsert', 'records': [{'id': '1', 'book': None}]}
        helpers.call_action('datastore_upsert', **data)
        search_result = _search(resource['id'])
        assert search_result['total'] == 1
        assert search_result['records'][0]['book'] is None
        assert search_result['records'][0]['author'] == 'Torres'

    @pytest.mark.ckan_config('ckan.plugins', 'datastore')
    @pytest.mark.usefixtures('clean_datastore', 'with_plugins')
    def test_upsert_doesnt_crash_with_json_field(self):
        if False:
            for i in range(10):
                print('nop')
        resource = factories.Resource()
        data = {'resource_id': resource['id'], 'force': True, 'primary_key': 'id', 'fields': [{'id': 'id', 'type': 'text'}, {'id': 'book', 'type': 'json'}, {'id': 'author', 'type': 'text'}]}
        helpers.call_action('datastore_create', **data)
        data = {'resource_id': resource['id'], 'force': True, 'method': 'insert', 'records': [{'id': '1', 'book': {'code': 'A', 'title': u'ñ'}, 'author': 'tolstoy'}]}
        helpers.call_action('datastore_upsert', **data)

    @pytest.mark.ckan_config('ckan.plugins', 'datastore')
    @pytest.mark.usefixtures('clean_datastore', 'with_plugins')
    def test_upsert_doesnt_crash_with_json_field_with_string_value(self):
        if False:
            for i in range(10):
                print('nop')
        resource = factories.Resource()
        data = {'resource_id': resource['id'], 'force': True, 'primary_key': 'id', 'fields': [{'id': 'id', 'type': 'text'}, {'id': 'book', 'type': 'json'}, {'id': 'author', 'type': 'text'}]}
        helpers.call_action('datastore_create', **data)
        data = {'resource_id': resource['id'], 'force': True, 'method': 'insert', 'records': [{'id': '1', 'book': u'ñ', 'author': 'tolstoy'}]}
        helpers.call_action('datastore_upsert', **data)

    @pytest.mark.ckan_config('ckan.plugins', 'datastore')
    @pytest.mark.usefixtures('clean_datastore', 'with_plugins')
    def test_dry_run(self):
        if False:
            print('Hello World!')
        ds = factories.Dataset()
        table = helpers.call_action(u'datastore_create', resource={u'package_id': ds['id']}, fields=[{u'id': u'spam', u'type': u'text'}], primary_key=u'spam')
        helpers.call_action(u'datastore_upsert', resource_id=table['resource_id'], records=[{u'spam': u'SPAM'}, {u'spam': u'EGGS'}], dry_run=True)
        result = helpers.call_action(u'datastore_search', resource_id=table['resource_id'])
        assert result['records'] == []

    @pytest.mark.ckan_config('ckan.plugins', 'datastore')
    @pytest.mark.usefixtures('clean_datastore', 'with_plugins')
    def test_dry_run_type_error(self):
        if False:
            i = 10
            return i + 15
        ds = factories.Dataset()
        table = helpers.call_action(u'datastore_create', resource={u'package_id': ds['id']}, fields=[{u'id': u'spam', u'type': u'numeric'}], primary_key=u'spam')
        try:
            helpers.call_action(u'datastore_upsert', resource_id=table['resource_id'], records=[{u'spam': u'SPAM'}, {u'spam': u'EGGS'}], dry_run=True)
        except ValidationError as ve:
            assert ve.error_dict['records'] == [u'invalid input syntax for type numeric: "SPAM"']
        else:
            assert 0, 'error not raised'

    @pytest.mark.ckan_config('ckan.plugins', 'datastore')
    @pytest.mark.usefixtures('clean_datastore', 'with_plugins')
    def test_dry_run_trigger_error(self):
        if False:
            print('Hello World!')
        ds = factories.Dataset()
        helpers.call_action(u'datastore_function_create', name=u'spamexception_trigger', rettype=u'trigger', definition=u'\n                BEGIN\n                IF NEW.spam != \'spam\' THEN\n                    RAISE EXCEPTION \'"%"? Yeeeeccch!\', NEW.spam;\n                END IF;\n                RETURN NEW;\n                END;')
        table = helpers.call_action(u'datastore_create', resource={u'package_id': ds['id']}, fields=[{u'id': u'spam', u'type': u'text'}], primary_key=u'spam', triggers=[{u'function': u'spamexception_trigger'}])
        try:
            helpers.call_action(u'datastore_upsert', resource_id=table['resource_id'], records=[{u'spam': u'EGGS'}], dry_run=True)
        except ValidationError as ve:
            assert ve.error_dict['records'] == [u'"EGGS"? Yeeeeccch!']
        else:
            assert 0, 'error not raised'

    @pytest.mark.ckan_config('ckan.plugins', 'datastore')
    @pytest.mark.usefixtures('clean_datastore', 'with_plugins')
    def test_calculate_record_count_is_false(self):
        if False:
            for i in range(10):
                print('nop')
        resource = factories.Resource()
        data = {'resource_id': resource['id'], 'force': True, 'fields': [{'id': 'name', 'type': 'text'}, {'id': 'age', 'type': 'text'}]}
        helpers.call_action('datastore_create', **data)
        data = {'resource_id': resource['id'], 'force': True, 'method': 'insert', 'records': [{'name': 'Sunita', 'age': '51'}, {'name': 'Bowan', 'age': '68'}]}
        helpers.call_action('datastore_upsert', **data)
        last_analyze = when_was_last_analyze(resource['id'])
        assert last_analyze is None

    @pytest.mark.ckan_config('ckan.plugins', 'datastore')
    @pytest.mark.usefixtures('clean_datastore', 'with_plugins')
    @pytest.mark.flaky(reruns=2)
    def test_calculate_record_count(self):
        if False:
            for i in range(10):
                print('nop')
        resource = factories.Resource()
        data = {'resource_id': resource['id'], 'force': True, 'fields': [{'id': 'name', 'type': 'text'}, {'id': 'age', 'type': 'text'}]}
        helpers.call_action('datastore_create', **data)
        data = {'resource_id': resource['id'], 'force': True, 'method': 'insert', 'records': [{'name': 'Sunita', 'age': '51'}, {'name': 'Bowan', 'age': '68'}], 'calculate_record_count': True}
        helpers.call_action('datastore_upsert', **data)
        last_analyze = when_was_last_analyze(resource['id'])
        assert last_analyze is not None

    @pytest.mark.ckan_config('ckan.plugins', 'datastore')
    @pytest.mark.usefixtures('clean_datastore', 'with_plugins')
    def test_no_pk_update(self):
        if False:
            print('Hello World!')
        resource = factories.Resource()
        data = {'resource_id': resource['id'], 'force': True, 'fields': [{'id': 'book', 'type': 'text'}], 'records': [{'book': u'El Niño'}]}
        helpers.call_action('datastore_create', **data)
        data = {'resource_id': resource['id'], 'force': True, 'method': 'upsert', 'records': [{'_id': '1', 'book': u'The boy'}]}
        helpers.call_action('datastore_upsert', **data)
        search_result = _search(resource['id'])
        assert search_result['total'] == 1
        assert search_result['records'][0]['book'] == 'The boy'

    @pytest.mark.ckan_config('ckan.plugins', 'datastore')
    @pytest.mark.usefixtures('clean_datastore', 'with_plugins')
    def test_id_instead_of_pk_update(self):
        if False:
            print('Hello World!')
        resource = factories.Resource()
        data = {'resource_id': resource['id'], 'force': True, 'primary_key': 'pk', 'fields': [{'id': 'pk', 'type': 'text'}, {'id': 'book', 'type': 'text'}, {'id': 'author', 'type': 'text'}], 'records': [{'pk': '1000', 'book': u'El Niño', 'author': 'Torres'}]}
        helpers.call_action('datastore_create', **data)
        data = {'resource_id': resource['id'], 'force': True, 'method': 'upsert', 'records': [{'_id': '1', 'book': u'The boy', 'author': u'F Torres'}]}
        helpers.call_action('datastore_upsert', **data)
        search_result = _search(resource['id'])
        assert search_result['total'] == 1
        assert search_result['records'][0]['pk'] == '1000'
        assert search_result['records'][0]['book'] == 'The boy'
        assert search_result['records'][0]['author'] == 'F Torres'

    @pytest.mark.ckan_config('ckan.plugins', 'datastore')
    @pytest.mark.usefixtures('clean_datastore', 'with_plugins')
    def test_empty_string_instead_of_null(self):
        if False:
            i = 10
            return i + 15
        resource = factories.Resource()
        data = {'resource_id': resource['id'], 'force': True, 'primary_key': 'pk', 'fields': [{'id': 'pk', 'type': 'text'}, {'id': 'n', 'type': 'int'}, {'id': 'd', 'type': 'date'}], 'records': [{'pk': '1000', 'n': '5', 'd': '2020-02-02'}]}
        helpers.call_action('datastore_create', **data)
        data = {'resource_id': resource['id'], 'force': True, 'method': 'upsert', 'records': [{'pk': '1000', 'n': '', 'd': ''}]}
        helpers.call_action('datastore_upsert', **data)
        search_result = _search(resource['id'])
        assert search_result['total'] == 1
        rec = search_result['records'][0]
        assert rec == {'_id': 1, 'pk': '1000', 'n': None, 'd': None}

class TestDatastoreInsert(object):

    @pytest.mark.ckan_config('ckan.plugins', 'datastore')
    @pytest.mark.usefixtures('clean_datastore', 'with_plugins')
    def test_basic_insert(self):
        if False:
            return 10
        resource = factories.Resource()
        data = {'resource_id': resource['id'], 'force': True, 'primary_key': 'id', 'fields': [{'id': 'id', 'type': 'text'}, {'id': 'book', 'type': 'text'}, {'id': 'author', 'type': 'text'}]}
        helpers.call_action('datastore_create', **data)
        data = {'resource_id': resource['id'], 'force': True, 'method': 'insert', 'records': [{'id': '1', 'book': u'El Niño', 'author': 'Torres'}]}
        helpers.call_action('datastore_upsert', **data)
        search_result = _search(resource['id'])
        assert search_result['total'] == 1
        assert search_result['fields'] == [{u'id': '_id', u'type': 'int'}, {u'id': u'id', u'type': u'text'}, {u'id': u'book', u'type': u'text'}, {u'id': u'author', u'type': u'text'}]
        assert search_result['records'][0] == {u'book': u'El Niño', u'_id': 1, u'id': u'1', u'author': u'Torres'}

    @pytest.mark.ckan_config('ckan.plugins', 'datastore')
    @pytest.mark.usefixtures('clean_datastore', 'with_plugins')
    def test_non_existing_field(self):
        if False:
            print('Hello World!')
        resource = factories.Resource(url_type='datastore')
        data = {'resource_id': resource['id'], 'force': True, 'primary_key': u'id', 'fields': [{'id': 'id', 'type': 'text'}, {'id': 'book', 'type': 'text'}, {'id': 'author', 'type': 'text'}], 'records': []}
        helpers.call_action('datastore_create', **data)
        data = {'resource_id': resource['id'], 'method': 'insert', 'records': [{'id': '1', 'dummy': 'tolkien'}]}
        with pytest.raises(ValidationError) as context:
            helpers.call_action('datastore_upsert', **data)
        assert u'row "1" has extra keys "dummy"' in str(context.value)

    @pytest.mark.ckan_config('ckan.plugins', 'datastore')
    @pytest.mark.usefixtures('clean_datastore', 'with_plugins')
    def test_key_already_exists(self):
        if False:
            while True:
                i = 10
        resource = factories.Resource()
        data = {'resource_id': resource['id'], 'force': True, 'primary_key': 'id', 'fields': [{'id': 'id', 'type': 'text'}, {'id': 'book', 'type': 'text'}, {'id': 'author', 'type': 'text'}], 'records': [{'id': '1', 'book': 'guide', 'author': 'adams'}]}
        helpers.call_action('datastore_create', **data)
        data = {'resource_id': resource['id'], 'force': True, 'method': 'insert', 'records': [{'id': '1', 'book': u'El Niño', 'author': 'Torres'}]}
        with pytest.raises(ValidationError) as context:
            helpers.call_action('datastore_upsert', **data)
        assert u'duplicate key value violates unique constraint' in str(context.value)

    @pytest.mark.ckan_config('ckan.plugins', 'datastore')
    @pytest.mark.usefixtures('clean_datastore', 'with_plugins')
    def test_empty_string_instead_of_null(self):
        if False:
            return 10
        resource = factories.Resource()
        data = {'resource_id': resource['id'], 'force': True, 'primary_key': 'pk', 'fields': [{'id': 'pk', 'type': 'text'}, {'id': 'n', 'type': 'int'}, {'id': 'd', 'type': 'date'}]}
        helpers.call_action('datastore_create', **data)
        data = {'resource_id': resource['id'], 'force': True, 'method': 'insert', 'records': [{'pk': '1000', 'n': '', 'd': ''}]}
        helpers.call_action('datastore_upsert', **data)
        search_result = _search(resource['id'])
        assert search_result['total'] == 1
        rec = search_result['records'][0]
        assert rec == {'_id': 1, 'pk': '1000', 'n': None, 'd': None}

    @pytest.mark.ckan_config('ckan.plugins', 'datastore')
    @pytest.mark.usefixtures('clean_datastore', 'with_plugins')
    def test_insert_wrong_type(self):
        if False:
            return 10
        resource = factories.Resource()
        data = {'resource_id': resource['id'], 'force': True, 'fields': [{'id': 'num', 'type': 'int'}]}
        helpers.call_action('datastore_create', **data)
        data = {'resource_id': resource['id'], 'force': True, 'method': 'insert', 'records': [{'num': 'notanumber'}]}
        with pytest.raises(ValidationError) as context:
            helpers.call_action('datastore_upsert', **data)
        assert u'invalid input syntax for integer: "notanumber"' in str(context.value)

class TestDatastoreUpdate(object):

    @pytest.mark.ckan_config('ckan.plugins', 'datastore')
    @pytest.mark.usefixtures('clean_datastore', 'with_plugins')
    def test_basic(self):
        if False:
            return 10
        resource = factories.Resource(url_type='datastore')
        data = {'resource_id': resource['id'], 'force': True, 'primary_key': u'id', 'fields': [{'id': 'id', 'type': 'text'}, {'id': 'book', 'type': 'text'}, {'id': 'author', 'type': 'text'}], 'records': [{'id': '1', 'book': u'El Niño', 'author': 'Torres'}]}
        helpers.call_action('datastore_create', **data)
        data = {'resource_id': resource['id'], 'method': 'update', 'records': [{'id': '1', 'book': u'The boy'}]}
        helpers.call_action('datastore_upsert', **data)
        search_result = _search(resource['id'])
        assert search_result['total'] == 1
        assert search_result['records'][0]['book'] == 'The boy'
        assert search_result['records'][0]['author'] == 'Torres'

    @pytest.mark.ckan_config('ckan.plugins', 'datastore')
    @pytest.mark.usefixtures('clean_datastore', 'with_plugins')
    def test_field_types(self):
        if False:
            print('Hello World!')
        resource = factories.Resource(url_type='datastore')
        data = {'resource_id': resource['id'], 'force': True, 'primary_key': u'bük', 'fields': [{'id': u'bük', 'type': 'text'}, {'id': 'author', 'type': 'text'}, {'id': 'nested', 'type': 'json'}, {'id': 'characters', 'type': 'text[]'}, {'id': 'published'}], 'records': [{u'bük': 'annakarenina', 'author': 'tolstoy', 'published': '2005-03-01', 'nested': ['b', {'moo': 'moo'}]}, {u'bük': 'warandpeace', 'author': 'tolstoy', 'nested': {'a': 'b'}}, {'author': 'adams', 'characters': ['Arthur', 'Marvin'], 'nested': {'foo': 'bar'}, u'bük': u'guide to the galaxy'}]}
        helpers.call_action('datastore_create', **data)
        data = {'resource_id': resource['id'], 'method': 'update', 'records': [{'author': 'adams', 'characters': ['Bob', 'Marvin'], 'nested': {'baz': 3}, u'bük': u'guide to the galaxy'}]}
        helpers.call_action('datastore_upsert', **data)
        search_result = _search(resource['id'])
        assert search_result['total'] == 3
        assert search_result['records'][2]['author'] == 'adams'
        assert search_result['records'][2]['characters'] == ['Bob', 'Marvin']
        assert search_result['records'][2]['nested'] == {'baz': 3}

    @pytest.mark.ckan_config('ckan.plugins', 'datastore')
    @pytest.mark.usefixtures('clean_datastore', 'with_plugins')
    def test_update_unspecified_key(self):
        if False:
            return 10
        resource = factories.Resource(url_type='datastore')
        data = {'resource_id': resource['id'], 'force': True, 'primary_key': u'id', 'fields': [{'id': 'id', 'type': 'text'}, {'id': 'book', 'type': 'text'}, {'id': 'author', 'type': 'text'}], 'records': []}
        helpers.call_action('datastore_create', **data)
        data = {'resource_id': resource['id'], 'method': 'update', 'records': [{'author': 'tolkien'}]}
        with pytest.raises(ValidationError) as context:
            helpers.call_action('datastore_upsert', **data)
        assert u'fields "id" are missing' in str(context.value)

    @pytest.mark.ckan_config('ckan.plugins', 'datastore')
    @pytest.mark.usefixtures('clean_datastore', 'with_plugins')
    def test_update_unknown_key(self):
        if False:
            return 10
        resource = factories.Resource(url_type='datastore')
        data = {'resource_id': resource['id'], 'force': True, 'primary_key': u'id', 'fields': [{'id': 'id', 'type': 'text'}, {'id': 'book', 'type': 'text'}, {'id': 'author', 'type': 'text'}], 'records': []}
        helpers.call_action('datastore_create', **data)
        data = {'resource_id': resource['id'], 'method': 'update', 'records': [{'id': '1', 'author': 'tolkien'}]}
        with pytest.raises(ValidationError) as context:
            helpers.call_action('datastore_upsert', **data)
        assert u'key "[\\\'1\\\']" not found' in str(context.value)

    @pytest.mark.ckan_config('ckan.plugins', 'datastore')
    @pytest.mark.usefixtures('clean_datastore', 'with_plugins')
    def test_update_non_existing_field(self):
        if False:
            for i in range(10):
                print('nop')
        resource = factories.Resource(url_type='datastore')
        data = {'resource_id': resource['id'], 'force': True, 'primary_key': u'id', 'fields': [{'id': 'id', 'type': 'text'}, {'id': 'book', 'type': 'text'}, {'id': 'author', 'type': 'text'}], 'records': [{'id': '1', 'book': 'guide'}]}
        helpers.call_action('datastore_create', **data)
        data = {'resource_id': resource['id'], 'method': 'update', 'records': [{'id': '1', 'dummy': 'tolkien'}]}
        with pytest.raises(ValidationError) as context:
            helpers.call_action('datastore_upsert', **data)
        assert u'fields "dummy" do not exist' in str(context.value)

    @pytest.mark.ckan_config('ckan.plugins', 'datastore')
    @pytest.mark.usefixtures('clean_datastore', 'with_plugins')
    def test_no_pk_update(self):
        if False:
            print('Hello World!')
        resource = factories.Resource()
        data = {'resource_id': resource['id'], 'force': True, 'fields': [{'id': 'book', 'type': 'text'}], 'records': [{'book': u'El Niño'}]}
        helpers.call_action('datastore_create', **data)
        data = {'resource_id': resource['id'], 'force': True, 'method': 'update', 'records': [{'_id': '1', 'book': u'The boy'}]}
        helpers.call_action('datastore_upsert', **data)
        search_result = _search(resource['id'])
        assert search_result['total'] == 1
        assert search_result['records'][0]['book'] == 'The boy'

    @pytest.mark.ckan_config('ckan.plugins', 'datastore')
    @pytest.mark.usefixtures('clean_datastore', 'with_plugins')
    def test_id_instead_of_pk_update(self):
        if False:
            for i in range(10):
                print('nop')
        resource = factories.Resource()
        data = {'resource_id': resource['id'], 'force': True, 'primary_key': 'pk', 'fields': [{'id': 'pk', 'type': 'text'}, {'id': 'book', 'type': 'text'}, {'id': 'author', 'type': 'text'}], 'records': [{'pk': '1000', 'book': u'El Niño', 'author': 'Torres'}]}
        helpers.call_action('datastore_create', **data)
        data = {'resource_id': resource['id'], 'force': True, 'method': 'update', 'records': [{'_id': '1', 'book': u'The boy', 'author': u'F Torres'}]}
        helpers.call_action('datastore_upsert', **data)
        search_result = _search(resource['id'])
        assert search_result['total'] == 1
        assert search_result['records'][0]['pk'] == '1000'
        assert search_result['records'][0]['book'] == 'The boy'
        assert search_result['records'][0]['author'] == 'F Torres'

    @pytest.mark.ckan_config('ckan.plugins', 'datastore')
    @pytest.mark.usefixtures('clean_datastore', 'with_plugins')
    def test_empty_string_instead_of_null(self):
        if False:
            while True:
                i = 10
        resource = factories.Resource()
        data = {'resource_id': resource['id'], 'force': True, 'primary_key': 'pk', 'fields': [{'id': 'pk', 'type': 'text'}, {'id': 'n', 'type': 'int'}, {'id': 'd', 'type': 'date'}], 'records': [{'pk': '1000', 'n': '5', 'd': '2020-02-02'}]}
        helpers.call_action('datastore_create', **data)
        data = {'resource_id': resource['id'], 'force': True, 'method': 'update', 'records': [{'pk': '1000', 'n': '', 'd': ''}]}
        helpers.call_action('datastore_upsert', **data)
        search_result = _search(resource['id'])
        assert search_result['total'] == 1
        rec = search_result['records'][0]
        assert rec == {'_id': 1, 'pk': '1000', 'n': None, 'd': None}