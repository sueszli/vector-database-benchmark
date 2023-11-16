from coalib.parsing.filters.decorators import typed_filter

@typed_filter('bearclass')
def language_filter(bear, args):
    if False:
        for i in range(10):
            print('nop')
    '\n    Filters the bears by ``LANGUAGES``.\n\n    :param bear: Bear object.\n    :param args: Set of languages on which ``bear`` is to be filtered.\n    :return:     ``True`` if this bear matches the criteria inside args,\n                 ``False`` otherwise.\n    '
    return bool({lang.lower() for lang in bear.LANGUAGES} & (args | {'all'}))