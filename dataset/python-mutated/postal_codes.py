import re
from typing import Any, Optional
from . import PostalCodeRuleInclusionType

def group_values(pattern, *values):
    if False:
        i = 10
        return i + 15
    result: list[Optional[tuple[Any, ...]]] = []
    for value in values:
        try:
            val = re.match(pattern, value)
        except TypeError:
            result.append(None)
        else:
            result.append(val.groups() if val else None)
    return result

def cast_tuple_index_to_type(index, target_type, *tuples):
    if False:
        for i in range(10):
            print('nop')
    'Cast tuple index to type.\n\n    Return list of tuples same as received but with index item casted to\n    tagret_type.\n    '
    result = []
    for _tuple in tuples:
        to_result = []
        try:
            for (i, entry) in enumerate(_tuple):
                to_result.append(entry if i != index else target_type(entry))
        except TypeError:
            pass
        result.append(tuple(to_result))
    return result

def compare_values(code, start, end):
    if False:
        for i in range(10):
            print('nop')
    if not code or not start:
        return False
    if not end:
        return start <= code
    return start <= code <= end

def check_uk_postal_code(code, start, end):
    if False:
        for i in range(10):
            print('nop')
    'Check postal code for uk, split the code by regex.\n\n    Example postal codes: BH20 2BC  (UK), IM16 7HF  (Isle of Man).\n    '
    pattern = '^([A-Z]{1,2})([0-9]+)([A-Z]?) ?([0-9][A-Z]{2})$'
    (code, start, end) = group_values(pattern, code, start, end)
    (code, start, end) = cast_tuple_index_to_type(1, int, code, start, end)
    return compare_values(code, start, end)

def check_irish_postal_code(code, start, end):
    if False:
        return 10
    'Check postal code for Ireland, split the code by regex.\n\n    Example postal codes: A65 2F0A, A61 2F0G.\n    '
    pattern = '([\\dA-Z]{3}) ?([\\dA-Z]{4})'
    (code, start, end) = group_values(pattern, code, start, end)
    return compare_values(code, start, end)

def check_any_postal_code(code, start, end):
    if False:
        for i in range(10):
            print('nop')
    'Fallback for any country not present in country_func_map.\n\n    Perform simple lexicographical comparison without splitting to sections.\n    '
    return compare_values(code, start, end)

def check_postal_code_in_range(country, code, start, end):
    if False:
        return 10
    country_func_map = {'GB': check_uk_postal_code, 'IM': check_uk_postal_code, 'GG': check_uk_postal_code, 'JE': check_uk_postal_code, 'IE': check_irish_postal_code}
    return country_func_map.get(country, check_any_postal_code)(code, start, end)

def check_shipping_method_for_postal_code(customer_shipping_address, method):
    if False:
        print('Hello World!')
    country = customer_shipping_address.country.code
    postal_code = customer_shipping_address.postal_code
    postal_code_rules = method.postal_code_rules.all()
    return {rule: check_postal_code_in_range(country, postal_code, rule.start, rule.end) for rule in postal_code_rules}

def is_shipping_method_applicable_for_postal_code(customer_shipping_address, method) -> bool:
    if False:
        while True:
            i = 10
    'Return if shipping method is applicable with the postal code rules.'
    results = check_shipping_method_for_postal_code(customer_shipping_address, method)
    if not results:
        return True
    if all(map(lambda rule: rule.inclusion_type == PostalCodeRuleInclusionType.INCLUDE, results.keys())):
        return any(results.values())
    if all(map(lambda rule: rule.inclusion_type == PostalCodeRuleInclusionType.EXCLUDE, results.keys())):
        return not any(results.values())
    return False

def filter_shipping_methods_by_postal_code_rules(shipping_methods, shipping_address):
    if False:
        for i in range(10):
            print('nop')
    'Filter shipping methods for given address by postal code rules.'
    excluded_methods_by_postal_code = []
    for method in shipping_methods:
        if not is_shipping_method_applicable_for_postal_code(shipping_address, method):
            excluded_methods_by_postal_code.append(method.pk)
    if excluded_methods_by_postal_code:
        return shipping_methods.exclude(pk__in=excluded_methods_by_postal_code)
    return shipping_methods