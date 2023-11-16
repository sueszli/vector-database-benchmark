"""Within this module we implement a *demo offline engine*.  Do not look to
close to the implementation, its just a simple example.  To get in use of this
*demo* engine add the following entry to your engines list in ``settings.yml``:

.. code:: yaml

  - name: my offline engine
    engine: demo_offline
    shortcut: demo
    disabled: false

"""
import json
engine_type = 'offline'
categories = ['general']
disabled = True
timeout = 2.0
about = {'wikidata_id': None, 'official_api_documentation': None, 'use_official_api': False, 'require_api_key': False, 'results': 'JSON'}
_my_offline_engine = None

def init(engine_settings=None):
    if False:
        print('Hello World!')
    'Initialization of the (offline) engine.  The origin of this demo engine is a\n    simple json string which is loaded in this example while the engine is\n    initialized.\n\n    '
    global _my_offline_engine
    _my_offline_engine = '[ {"value": "%s"}, {"value":"first item"}, {"value":"second item"}, {"value":"third item"}]' % engine_settings.get('name')

def search(query, request_params):
    if False:
        return 10
    "Query (offline) engine and return results.  Assemble the list of results from\n    your local engine.  In this demo engine we ignore the 'query' term, usual\n    you would pass the 'query' term to your local engine to filter out the\n    results.\n\n    "
    ret_val = []
    result_list = json.loads(_my_offline_engine)
    for row in result_list:
        entry = {'query': query, 'language': request_params['searxng_locale'], 'value': row.get('value'), 'template': 'key-value.html'}
        ret_val.append(entry)
    return ret_val