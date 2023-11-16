import textwrap
from .util import coins_to_satoshis, satoshis_to_coins

def lbc_to_dewies(lbc: str) -> int:
    if False:
        i = 10
        return i + 15
    try:
        return coins_to_satoshis(lbc)
    except ValueError:
        raise ValueError(textwrap.dedent(f"\n            Decimal inputs require a value in the ones place and in the tenths place\n            separated by a period. The value provided, '{lbc}', is not of the correct\n            format.\n\n            The following are examples of valid decimal inputs:\n\n            1.0\n            0.001\n            2.34500\n            4534.4\n            2323434.0000\n\n            The following are NOT valid:\n\n            83\n            .456\n            123.\n            "))

def dewies_to_lbc(dewies) -> str:
    if False:
        i = 10
        return i + 15
    return satoshis_to_coins(dewies)

def dict_values_to_lbc(d):
    if False:
        i = 10
        return i + 15
    lbc_dict = {}
    for (key, value) in d.items():
        if isinstance(value, int):
            lbc_dict[key] = dewies_to_lbc(value)
        elif isinstance(value, dict):
            lbc_dict[key] = dict_values_to_lbc(value)
        else:
            lbc_dict[key] = value
    return lbc_dict