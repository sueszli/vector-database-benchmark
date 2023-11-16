import warnings
from . import _arffread
__all__ = ['MetaData', 'loadarff', 'ArffError', 'ParseArffError', 'r_meta', 'r_comment', 'r_empty', 'r_headerline', 'r_datameta', 'r_relation', 'r_attribute', 'r_nominal', 'r_date', 'r_comattrval', 'r_wcomattrval', 'Attribute', 'NominalAttribute', 'NumericAttribute', 'StringAttribute', 'DateAttribute', 'RelationalAttribute', 'to_attribute', 'csv_sniffer_has_bug_last_field', 'workaround_csv_sniffer_bug_last_field', 'split_data_line', 'tokenize_attribute', 'tokenize_single_comma', 'tokenize_single_wcomma', 'read_relational_attribute', 'read_header', 'basic_stats', 'print_attribute', 'test_weka']

def __dir__():
    if False:
        for i in range(10):
            print('nop')
    return __all__

def __getattr__(name):
    if False:
        while True:
            i = 10
    if name not in __all__:
        raise AttributeError(f'scipy.io.arff.arffread is deprecated and has no attribute {name}. Try looking in scipy.io.arff instead.')
    warnings.warn(f'Please use `{name}` from the `scipy.io.arff` namespace, the `scipy.io.arff.arffread` namespace is deprecated.', category=DeprecationWarning, stacklevel=2)
    return getattr(_arffread, name)