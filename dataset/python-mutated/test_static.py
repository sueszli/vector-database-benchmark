import pytest
from warehouse.utils.static import ManifestCacheBuster

class TestManifestCacheBuster:

    def test_returns_when_valid(self, monkeypatch):
        if False:
            return 10
        monkeypatch.setattr(ManifestCacheBuster, 'get_manifest', lambda x: {'/the/path/style.css': '/the/busted/path/style.css'})
        cb = ManifestCacheBuster('warehouse:static/dist/manifest.json')
        result = cb(None, '/the/path/style.css', {'keyword': 'arg'})
        assert result == ('/the/busted/path/style.css', {'keyword': 'arg'})

    def test_raises_when_invalid(self, monkeypatch):
        if False:
            for i in range(10):
                print('nop')
        monkeypatch.setattr(ManifestCacheBuster, 'get_manifest', lambda x: {})
        cb = ManifestCacheBuster('warehouse:static/dist/manifest.json')
        with pytest.raises(ValueError):
            cb(None, '/the/path/style.css', {'keyword': 'arg'})

    def test_returns_when_invalid_and_not_strict(self, monkeypatch):
        if False:
            print('Hello World!')
        monkeypatch.setattr(ManifestCacheBuster, 'get_manifest', lambda x: {})
        cb = ManifestCacheBuster('warehouse:static/dist/manifest.json', strict=False)
        result = cb(None, '/the/path/style.css', {'keyword': 'arg'})
        assert result == ('/the/path/style.css', {'keyword': 'arg'})