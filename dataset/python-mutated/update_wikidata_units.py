"""Fetch units from :origin:`searx/engines/wikidata.py` engine.

Output file: :origin:`searx/data/wikidata_units.json` (:origin:`CI Update data
...  <.github/workflows/data-update.yml>`).

"""
import json
import collections
from os.path import join
from searx import searx_dir
from searx.engines import wikidata, set_loggers
set_loggers(wikidata, 'wikidata')
SARQL_REQUEST = '\nSELECT DISTINCT ?item ?symbol\nWHERE\n{\n  ?item wdt:P31/wdt:P279 wd:Q47574 .\n  ?item p:P5061 ?symbolP .\n  ?symbolP ps:P5061 ?symbol ;\n           wikibase:rank ?rank .\n  FILTER(LANG(?symbol) = "en").\n}\nORDER BY ?item DESC(?rank) ?symbol\n'

def get_data():
    if False:
        for i in range(10):
            print('nop')
    results = collections.OrderedDict()
    response = wikidata.send_wikidata_query(SARQL_REQUEST)
    for unit in response['results']['bindings']:
        name = unit['item']['value'].replace('http://www.wikidata.org/entity/', '')
        unit = unit['symbol']['value']
        if name not in results:
            results[name] = unit
    return results

def get_wikidata_units_filename():
    if False:
        for i in range(10):
            print('nop')
    return join(join(searx_dir, 'data'), 'wikidata_units.json')
if __name__ == '__main__':
    with open(get_wikidata_units_filename(), 'w', encoding='utf8') as f:
        json.dump(get_data(), f, indent=4, ensure_ascii=False)