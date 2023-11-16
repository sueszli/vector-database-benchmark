from coalib.parsing.filters.decorators import typed_filter

@typed_filter('bearclass')
def can_detect_filter(bear, args):
    if False:
        print('Hello World!')
    '\n    Filters the bears by ``CAN_DETECT``.\n\n    :param bear: Bear object.\n    :param args: Set of detectable issue types on which ``bear`` is to be\n                 filtered.\n    :return:     ``True`` if this bear matches the criteria inside args,\n                 ``False`` otherwise.\n    '
    return bool({detect.lower() for detect in bear.CAN_DETECT} & args)