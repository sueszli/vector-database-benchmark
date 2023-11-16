from __future__ import annotations
import json
import os
from flask import url_for

def configure_manifest_files(app):
    if False:
        print('Hello World!')
    'Load the manifest file and register the `url_for_asset_` template tag.\n\n    :param app:\n    '
    manifest = {}

    def parse_manifest_json():
        if False:
            while True:
                i = 10
        try:
            manifest_file = os.path.join(os.path.dirname(__file__), os.pardir, 'static/dist/manifest.json')
            with open(manifest_file) as file:
                manifest.update(json.load(file))
                for (source, target) in manifest.copy().items():
                    manifest[source] = os.path.join('dist', target)
        except Exception:
            print('Please make sure to build the frontend in static/ directory and restart the server')

    def get_asset_url(filename):
        if False:
            i = 10
            return i + 15
        if app.debug:
            parse_manifest_json()
        return url_for('static', filename=manifest.get(filename, filename))
    parse_manifest_json()

    @app.context_processor
    def get_url_for_asset():
        if False:
            for i in range(10):
                print('nop')
        'Template tag to return the asset URL.\n\n        WebPack renders the assets after minification and modification under the\n        static/dist folder. This template tag reads the asset name in\n        ``manifest.json`` and returns the appropriate file.\n        '
        return {'url_for_asset': get_asset_url}