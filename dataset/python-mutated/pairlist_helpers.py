import re
from typing import List
from freqtrade.constants import Config

def expand_pairlist(wildcardpl: List[str], available_pairs: List[str], keep_invalid: bool=False) -> List[str]:
    if False:
        print('Hello World!')
    "\n    Expand pairlist potentially containing wildcards based on available markets.\n    This will implicitly filter all pairs in the wildcard-list which are not in available_pairs.\n    :param wildcardpl: List of Pairlists, which may contain regex\n    :param available_pairs: List of all available pairs (`exchange.get_markets().keys()`)\n    :param keep_invalid: If sets to True, drops invalid pairs silently while expanding regexes\n    :return: expanded pairlist, with Regexes from wildcardpl applied to match all available pairs.\n    :raises: ValueError if a wildcard is invalid (like '*/BTC' - which should be `.*/BTC`)\n    "
    result = []
    if keep_invalid:
        for pair_wc in wildcardpl:
            try:
                comp = re.compile(pair_wc, re.IGNORECASE)
                result_partial = [pair for pair in available_pairs if re.fullmatch(comp, pair)]
                result += result_partial or [pair_wc]
            except re.error as err:
                raise ValueError(f'Wildcard error in {pair_wc}, {err}')
        result = [element for element in result if re.fullmatch('^[A-Za-z0-9:/-]+$', element)]
    else:
        for pair_wc in wildcardpl:
            try:
                comp = re.compile(pair_wc, re.IGNORECASE)
                result += [pair for pair in available_pairs if re.fullmatch(comp, pair)]
            except re.error as err:
                raise ValueError(f'Wildcard error in {pair_wc}, {err}')
    return result

def dynamic_expand_pairlist(config: Config, markets: List[str]) -> List[str]:
    if False:
        while True:
            i = 10
    expanded_pairs = expand_pairlist(config['pairs'], markets)
    if config.get('freqai', {}).get('enabled', False):
        corr_pairlist = config['freqai']['feature_parameters']['include_corr_pairlist']
        expanded_pairs += [pair for pair in corr_pairlist if pair not in config['pairs']]
    return expanded_pairs