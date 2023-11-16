"""Processors for engine-type: ``online_currency``

"""
import unicodedata
import re
from searx.data import CURRENCIES
from .online import OnlineProcessor
parser_re = re.compile('.*?(\\d+(?:\\.\\d+)?) ([^.0-9]+) (?:in|to) ([^.0-9]+)', re.I)

def normalize_name(name):
    if False:
        while True:
            i = 10
    name = name.lower().replace('-', ' ').rstrip('s')
    name = re.sub(' +', ' ', name)
    return unicodedata.normalize('NFKD', name).lower()

def name_to_iso4217(name):
    if False:
        i = 10
        return i + 15
    name = normalize_name(name)
    currency = CURRENCIES['names'].get(name, [name])
    if isinstance(currency, str):
        return currency
    return currency[0]

def iso4217_to_name(iso4217, language):
    if False:
        while True:
            i = 10
    return CURRENCIES['iso4217'].get(iso4217, {}).get(language, iso4217)

class OnlineCurrencyProcessor(OnlineProcessor):
    """Processor class used by ``online_currency`` engines."""
    engine_type = 'online_currency'

    def get_params(self, search_query, engine_category):
        if False:
            print('Hello World!')
        'Returns a set of :ref:`request params <engine request online_currency>`\n        or ``None`` if search query does not match to :py:obj:`parser_re`.'
        params = super().get_params(search_query, engine_category)
        if params is None:
            return None
        m = parser_re.match(search_query.query)
        if not m:
            return None
        (amount_str, from_currency, to_currency) = m.groups()
        try:
            amount = float(amount_str)
        except ValueError:
            return None
        from_currency = name_to_iso4217(from_currency.strip())
        to_currency = name_to_iso4217(to_currency.strip())
        params['amount'] = amount
        params['from'] = from_currency
        params['to'] = to_currency
        params['from_name'] = iso4217_to_name(from_currency, 'en')
        params['to_name'] = iso4217_to_name(to_currency, 'en')
        return params

    def get_default_tests(self):
        if False:
            for i in range(10):
                print('nop')
        tests = {}
        tests['currency'] = {'matrix': {'query': '1337 usd in rmb'}, 'result_container': ['has_answer']}
        return tests