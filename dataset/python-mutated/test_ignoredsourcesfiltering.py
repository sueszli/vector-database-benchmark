import pytest
from sentry.lang.native.sources import filter_ignored_sources
from sentry.testutils.helpers import override_options
from sentry.testutils.pytest.fixtures import django_db_all

class TestIgnoredSourcesFiltering:

    @pytest.fixture
    def sources(self):
        if False:
            i = 10
            return i + 15
        builtins = [{'id': 'sentry:microsoft', 'name': 'Microsoft', 'type': 'gcs'}, {'id': 'sentry:electron', 'name': 'Electron', 'type': 's3'}, {'id': 'sentry:ios-source', 'name': 'iOS', 'type': 'http'}, {'id': 'sentry:tvos-source', 'name': 'iOS', 'type': 'http'}, {'type': 'http', 'id': 'custom', 'layout': {'type': 'symstore'}, 'url': 'https://msdl.microsoft.com/download/symbols/'}]
        return builtins

    @pytest.fixture
    def reversed_alias_map(self):
        if False:
            print('Hello World!')
        return {'sentry:ios-source': 'sentry:ios', 'sentry:tvos-source': 'sentry:ios'}

    @django_db_all
    def test_sources_included_and_ignored_empty(self):
        if False:
            while True:
                i = 10
        with override_options({'symbolicator.ignored_sources': []}):
            sources = filter_ignored_sources([])
            assert sources == []

    @django_db_all
    def test_sources_ignored_unset(self, sources):
        if False:
            while True:
                i = 10
        sources = filter_ignored_sources(sources)
        source_ids = list(map(lambda s: s['id'], sources))
        assert source_ids == ['sentry:microsoft', 'sentry:electron', 'sentry:ios-source', 'sentry:tvos-source', 'custom']

    @django_db_all
    def test_sources_ignored_empty(self, sources):
        if False:
            while True:
                i = 10
        with override_options({'symbolicator.ignored_sources': []}):
            sources = filter_ignored_sources(sources)
            source_ids = list(map(lambda s: s['id'], sources))
            assert source_ids == ['sentry:microsoft', 'sentry:electron', 'sentry:ios-source', 'sentry:tvos-source', 'custom']

    @django_db_all
    def test_sources_ignored_builtin(self, sources):
        if False:
            print('Hello World!')
        with override_options({'symbolicator.ignored_sources': ['sentry:microsoft']}):
            sources = filter_ignored_sources(sources)
            source_ids = list(map(lambda s: s['id'], sources))
            assert source_ids == ['sentry:electron', 'sentry:ios-source', 'sentry:tvos-source', 'custom']

    @django_db_all
    def test_sources_ignored_alias(self, sources, reversed_alias_map):
        if False:
            print('Hello World!')
        with override_options({'symbolicator.ignored_sources': ['sentry:ios']}):
            sources = filter_ignored_sources(sources, reversed_alias_map)
            source_ids = list(map(lambda s: s['id'], sources))
            assert source_ids == ['sentry:microsoft', 'sentry:electron', 'custom']

    @django_db_all
    def test_sources_ignored_bypass_alias(self, sources, reversed_alias_map):
        if False:
            print('Hello World!')
        with override_options({'symbolicator.ignored_sources': ['sentry:ios-source']}):
            sources = filter_ignored_sources(sources, reversed_alias_map)
            source_ids = list(map(lambda s: s['id'], sources))
            assert source_ids == ['sentry:microsoft', 'sentry:electron', 'sentry:tvos-source', 'custom']

    @django_db_all
    def test_sources_ignored_custom(self, sources):
        if False:
            for i in range(10):
                print('nop')
        with override_options({'symbolicator.ignored_sources': ['custom']}):
            sources = filter_ignored_sources(sources)
            source_ids = list(map(lambda s: s['id'], sources))
            assert source_ids == ['sentry:microsoft', 'sentry:electron', 'sentry:ios-source', 'sentry:tvos-source']

    @django_db_all
    def test_sources_ignored_unrecognized(self, sources):
        if False:
            i = 10
            return i + 15
        with override_options({'symbolicator.ignored_sources': ['honk']}):
            sources = filter_ignored_sources(sources)
            source_ids = list(map(lambda s: s['id'], sources))
            assert source_ids == ['sentry:microsoft', 'sentry:electron', 'sentry:ios-source', 'sentry:tvos-source', 'custom']