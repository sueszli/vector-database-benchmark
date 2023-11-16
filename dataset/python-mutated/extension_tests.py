from os.path import dirname
from unittest.mock import Mock
from superset.extensions import UIManifestProcessor
APP_DIR = f'{dirname(__file__)}/fixtures'

def test_get_manifest_with_prefix():
    if False:
        while True:
            i = 10
    app = Mock(config={'STATIC_ASSETS_PREFIX': 'https://cool.url/here'}, template_context_processors={None: []})
    manifest_processor = UIManifestProcessor(APP_DIR)
    manifest_processor.init_app(app)
    manifest = manifest_processor.get_manifest()
    assert manifest['js_manifest']('main') == ['/static/dist/main-js.js']
    assert manifest['css_manifest']('main') == ['/static/dist/main-css.css']
    assert manifest['js_manifest']('styles') == ['/static/dist/styles-js.js']
    assert manifest['css_manifest']('styles') == []
    assert manifest['assets_prefix'] == 'https://cool.url/here'

def test_get_manifest_no_prefix():
    if False:
        while True:
            i = 10
    app = Mock(config={'STATIC_ASSETS_PREFIX': ''}, template_context_processors={None: []})
    manifest_processor = UIManifestProcessor(APP_DIR)
    manifest_processor.init_app(app)
    manifest = manifest_processor.get_manifest()
    assert manifest['js_manifest']('main') == ['/static/dist/main-js.js']
    assert manifest['css_manifest']('main') == ['/static/dist/main-css.css']
    assert manifest['js_manifest']('styles') == ['/static/dist/styles-js.js']
    assert manifest['css_manifest']('styles') == []
    assert manifest['assets_prefix'] == ''