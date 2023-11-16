import enum
__all__ = ['QuantType']

class QuantType(enum.IntEnum):
    DYNAMIC = 0
    STATIC = 1
    QAT = 2
    WEIGHT_ONLY = 3
_quant_type_to_str = {QuantType.STATIC: 'static', QuantType.DYNAMIC: 'dynamic', QuantType.QAT: 'qat', QuantType.WEIGHT_ONLY: 'weight_only'}

def _get_quant_type_to_str(quant_type: QuantType) -> str:
    if False:
        print('Hello World!')
    return _quant_type_to_str[quant_type]

def _quant_type_from_str(name: str) -> QuantType:
    if False:
        print('Hello World!')
    for (quant_type, s) in _quant_type_to_str.items():
        if name == s:
            return quant_type
    raise ValueError(f"Unknown QuantType name '{name}'")