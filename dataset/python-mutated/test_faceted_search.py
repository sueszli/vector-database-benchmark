import pytest
from readthedocs.search.faceted_search import PageSearch

@pytest.mark.django_db
@pytest.mark.search
class TestPageSearch:

    @pytest.mark.parametrize('case', ['upper', 'lower', 'title'])
    def test_search_exact_match(self, client, project, case):
        if False:
            i = 10
            return i + 15
        'Check quoted query match exact phrase with case insensitively\n\n        Making a query with quoted text like ``"foo bar"`` should match\n        exactly ``foo bar`` or ``Foo Bar`` etc\n        '
        query_text = '"Sphinx uses"'
        cased_query = getattr(query_text, case)
        query = cased_query()
        page_search = PageSearch(query=query)
        results = page_search.execute()
        assert len(results) == 2
        assert results[0]['project'] == 'kuma'
        assert results[0]['path'] == 'testdocumentation'
        assert results[0]['version'] == 'latest'
        assert results[1]['project'] == 'kuma'
        assert results[1]['path'] == 'testdocumentation'
        assert results[1]['version'] == 'stable'

    def test_search_combined_result(self, client, project):
        if False:
            return 10
        'Check search result are combined of both `AND` and `OR` operator\n\n        If query is `Foo Bar` then the result should be as following order:\n\n        - Where both `Foo Bar` is present\n        - Where `Foo` or `Bar` is present\n        '
        query = 'Elasticsearch Query'
        page_search = PageSearch(query=query)
        results = page_search.execute()
        assert len(results) == 6
        result_paths_latest = [r.path for r in results if r.version == 'latest']
        result_paths_stable = [r.path for r in results if r.version == 'stable']
        expected_paths = ['guides/wipe-environment', 'docker', 'installation']
        assert result_paths_latest == expected_paths
        assert result_paths_stable == expected_paths