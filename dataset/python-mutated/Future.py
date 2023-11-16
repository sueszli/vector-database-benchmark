def _get_feature(name):
    if False:
        i = 10
        return i + 15
    import __future__
    return getattr(__future__, name, object())
unicode_literals = _get_feature('unicode_literals')
with_statement = _get_feature('with_statement')
division = _get_feature('division')
print_function = _get_feature('print_function')
absolute_import = _get_feature('absolute_import')
nested_scopes = _get_feature('nested_scopes')
generators = _get_feature('generators')
generator_stop = _get_feature('generator_stop')
annotations = _get_feature('annotations')
del _get_feature