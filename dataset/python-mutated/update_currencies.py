"""Fetch currencies from :origin:`searx/engines/wikidata.py` engine.

Output file: :origin:`searx/data/currencies.json` (:origin:`CI Update data ...
<.github/workflows/data-update.yml>`).

"""
import re
import unicodedata
import json
from os.path import join
from searx import searx_dir
from searx.locales import LOCALE_NAMES, locales_initialize
from searx.engines import wikidata, set_loggers
set_loggers(wikidata, 'wikidata')
locales_initialize()
SARQL_REQUEST = '\nSELECT DISTINCT ?iso4217 ?unit ?unicode ?label ?alias WHERE {\n  ?item wdt:P498 ?iso4217; rdfs:label ?label.\n  OPTIONAL { ?item skos:altLabel ?alias FILTER (LANG (?alias) = LANG(?label)). }\n  OPTIONAL { ?item wdt:P5061 ?unit. }\n  OPTIONAL { ?item wdt:P489 ?symbol.\n             ?symbol wdt:P487 ?unicode. }\n  MINUS { ?item wdt:P582 ?end_data . }                  # Ignore monney with an end date\n  MINUS { ?item wdt:P31/wdt:P279* wd:Q15893266 . }      # Ignore "former entity" (obsolete currency)\n  FILTER(LANG(?label) IN (%LANGUAGES_SPARQL%)).\n}\nORDER BY ?iso4217 ?unit ?unicode ?label ?alias\n'
SPARQL_WIKIPEDIA_NAMES_REQUEST = '\nSELECT DISTINCT ?iso4217 ?article_name WHERE {\n  ?item wdt:P498 ?iso4217 .\n  ?article schema:about ?item ;\n           schema:name ?article_name ;\n           schema:isPartOf [ wikibase:wikiGroup "wikipedia" ]\n  MINUS { ?item wdt:P582 ?end_data . }                  # Ignore monney with an end date\n  MINUS { ?item wdt:P31/wdt:P279* wd:Q15893266 . }      # Ignore "former entity" (obsolete currency)\n  FILTER(LANG(?article_name) IN (%LANGUAGES_SPARQL%)).\n}\nORDER BY ?iso4217 ?article_name\n'
LANGUAGES = LOCALE_NAMES.keys()
LANGUAGES_SPARQL = ', '.join(set(map(lambda l: repr(l.split('_')[0]), LANGUAGES)))

def remove_accents(name):
    if False:
        while True:
            i = 10
    return unicodedata.normalize('NFKD', name).lower()

def remove_extra(name):
    if False:
        for i in range(10):
            print('nop')
    for c in ('(', ':'):
        if c in name:
            name = name.split(c)[0].strip()
    return name

def _normalize_name(name):
    if False:
        i = 10
        return i + 15
    name = re.sub(' +', ' ', remove_accents(name.lower()).replace('-', ' '))
    name = remove_extra(name)
    return name

def add_currency_name(db, name, iso4217, normalize_name=True):
    if False:
        while True:
            i = 10
    db_names = db['names']
    if normalize_name:
        name = _normalize_name(name)
    iso4217_set = db_names.setdefault(name, [])
    if iso4217 not in iso4217_set:
        iso4217_set.insert(0, iso4217)

def add_currency_label(db, label, iso4217, language):
    if False:
        for i in range(10):
            print('nop')
    labels = db['iso4217'].setdefault(iso4217, {})
    labels[language] = label

def wikidata_request_result_iterator(request):
    if False:
        for i in range(10):
            print('nop')
    result = wikidata.send_wikidata_query(request.replace('%LANGUAGES_SPARQL%', LANGUAGES_SPARQL))
    if result is not None:
        for r in result['results']['bindings']:
            yield r

def fetch_db():
    if False:
        print('Hello World!')
    db = {'names': {}, 'iso4217': {}}
    for r in wikidata_request_result_iterator(SPARQL_WIKIPEDIA_NAMES_REQUEST):
        iso4217 = r['iso4217']['value']
        article_name = r['article_name']['value']
        article_lang = r['article_name']['xml:lang']
        add_currency_name(db, article_name, iso4217)
        add_currency_label(db, article_name, iso4217, article_lang)
    for r in wikidata_request_result_iterator(SARQL_REQUEST):
        iso4217 = r['iso4217']['value']
        if 'label' in r:
            label = r['label']['value']
            label_lang = r['label']['xml:lang']
            add_currency_name(db, label, iso4217)
            add_currency_label(db, label, iso4217, label_lang)
        if 'alias' in r:
            add_currency_name(db, r['alias']['value'], iso4217)
        if 'unicode' in r:
            add_currency_name(db, r['unicode']['value'], iso4217, normalize_name=False)
        if 'unit' in r:
            add_currency_name(db, r['unit']['value'], iso4217, normalize_name=False)
    return db

def get_filename():
    if False:
        return 10
    return join(join(searx_dir, 'data'), 'currencies.json')

def main():
    if False:
        while True:
            i = 10
    db = fetch_db()
    add_currency_name(db, 'euro', 'EUR')
    add_currency_name(db, 'euros', 'EUR')
    add_currency_name(db, 'dollar', 'USD')
    add_currency_name(db, 'dollars', 'USD')
    add_currency_name(db, 'peso', 'MXN')
    add_currency_name(db, 'pesos', 'MXN')
    for name in db['names']:
        if len(db['names'][name]) == 1:
            db['names'][name] = db['names'][name][0]
    with open(get_filename(), 'w', encoding='utf8') as f:
        json.dump(db, f, ensure_ascii=False, indent=4)
if __name__ == '__main__':
    main()