from sentry.api.serializers import serialize
from sentry.models.savedsearch import SavedSearch, Visibility
from sentry.testutils.cases import TestCase

class SavedSearchSerializerTest(TestCase):

    def test_simple(self):
        if False:
            print('Hello World!')
        search = SavedSearch.objects.create(name='Something', query='some query')
        result = serialize(search)
        assert result['id'] == str(search.id)
        assert result['type'] == search.type
        assert result['name'] == search.name
        assert result['query'] == search.query
        assert result['visibility'] == Visibility.OWNER
        assert result['dateCreated'] == search.date_added
        assert not result['isGlobal']
        assert not result['isPinned']

    def test_global(self):
        if False:
            while True:
                i = 10
        search = SavedSearch(name='Unresolved Issues', query='is:unresolved', is_global=True, visibility=Visibility.ORGANIZATION)
        result = serialize(search)
        assert result['id'] == str(search.id)
        assert result['type'] == search.type
        assert result['name'] == search.name
        assert result['query'] == search.query
        assert result['visibility'] == Visibility.ORGANIZATION
        assert result['dateCreated'] == search.date_added
        assert result['isGlobal']
        assert not result['isPinned']

    def test_organization(self):
        if False:
            i = 10
            return i + 15
        search = SavedSearch.objects.create(organization=self.organization, name='Something', query='some query', visibility=Visibility.ORGANIZATION)
        result = serialize(search)
        assert result['id'] == str(search.id)
        assert result['type'] == search.type
        assert result['name'] == search.name
        assert result['query'] == search.query
        assert result['visibility'] == Visibility.ORGANIZATION
        assert result['dateCreated'] == search.date_added
        assert not result['isGlobal']
        assert not result['isPinned']

    def test_pinned(self):
        if False:
            for i in range(10):
                print('nop')
        search = SavedSearch.objects.create(organization=self.organization, owner_id=self.user.id, name='Something', query='some query', visibility=Visibility.OWNER_PINNED)
        result = serialize(search)
        assert result['id'] == str(search.id)
        assert result['type'] == search.type
        assert result['name'] == search.name
        assert result['query'] == search.query
        assert result['visibility'] == Visibility.OWNER_PINNED
        assert result['dateCreated'] == search.date_added
        assert not result['isGlobal']
        assert result['isPinned']