__all__ = ['transform_to_bool']

def transform_to_bool(value):
    if False:
        for i in range(10):
            print('nop')
    "\n    Transforms a certain set of values to True or False.\n    True can be represented by '1', 'True' and 'true.'\n    False can be represented by '1', 'False' and 'false.'\n\n    Any other representation will be rejected.\n    "
    if value in ['1', 'true', 'True', True]:
        return True
    elif value in ['0', 'false', 'False', False]:
        return False
    raise ValueError('Invalid bool representation "%s" provided.' % value)