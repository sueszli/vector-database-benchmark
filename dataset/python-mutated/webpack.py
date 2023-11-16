import os
import simplejson
from flask import url_for
WEBPACK_MANIFEST_PATH = os.path.join(os.path.dirname(__file__), '../../client/dist/', 'asset-manifest.json')

def configure_webpack(app):
    if False:
        i = 10
        return i + 15
    app.extensions['webpack'] = {'assets': None}

    def get_asset(path):
        if False:
            i = 10
            return i + 15
        assets = app.extensions['webpack']['assets']
        if assets is None or app.debug:
            try:
                with open(WEBPACK_MANIFEST_PATH) as fp:
                    assets = simplejson.load(fp)
            except IOError:
                app.logger.exception('Unable to load webpack manifest')
                assets = {}
            app.extensions['webpack']['assets'] = assets
        return url_for('static', filename=assets.get(path, path))

    @app.context_processor
    def webpack_assets():
        if False:
            for i in range(10):
                print('nop')
        return {'asset_url': get_asset}