from coalib.parsing.filters.decorators import typed_filter

@typed_filter('bearclass')
def can_fix_filter(bear, args):
    if False:
        i = 10
        return i + 15
    '\n    Filters the bears by ``CAN_FIX``.\n\n    :param bear: Bear object.\n    :param args: Set of fixable issue types on which ``bear`` is to be\n                 filtered.\n    :return:     ``True`` if this bear matches the criteria inside args,\n                 ``False`` otherwise.\n    '
    return bool({fix.lower() for fix in bear.CAN_FIX} & args)