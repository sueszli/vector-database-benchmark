from collections import OrderedDict
from coalib.parsing.InvalidFilterException import InvalidFilterException
from coalib.parsing.filters import available_filters
from coalib.parsing.DefaultArgParser import default_arg_parser

def is_valid_filter(filter_name):
    if False:
        return 10
    return filter_name in available_filters

def _filter_section_bears(bears, args, filter_name):
    if False:
        while True:
            i = 10
    filter_function = available_filters[filter_name]
    return {section: tuple((bear for bear in bears[section] if filter_function(bear, args))) for section in bears}

def apply_filter(filter_name, filter_args, all_bears=None):
    if False:
        i = 10
        return i + 15
    "\n    Returns bears after filtering based on ``filter_args``. It returns\n    all bears if nothing is present in ``filter_args``.\n\n    :param filter_name:\n        Name of the filter.\n    :param filter_args:\n        Arguments of the filter to be passed in.\n        For example:\n        ``('c', 'java')``\n    :param all_bears:\n        List of bears on which filter is to be applied.\n        All the bears are loaded automatically by default.\n    :return:\n        Filtered bears based on a single filter.\n    "
    if all_bears is None:
        from coalib.settings.ConfigurationGathering import get_all_bears
        all_bears = get_all_bears()
    if not is_valid_filter(filter_name):
        raise InvalidFilterException(filter_name)
    if not filter_args or len(filter_args) == 0:
        return all_bears
    filter_args = {arg.lower() for arg in filter_args}
    (local_bears, global_bears) = all_bears
    local_bears = _filter_section_bears(local_bears, filter_args, filter_name)
    global_bears = _filter_section_bears(global_bears, filter_args, filter_name)
    return (local_bears, global_bears)

def apply_filters(filters, bears=None, sections=None):
    if False:
        while True:
            i = 10
    "\n    Returns bears or sections after filtering based on ``filters``.\n    It returns intersection if more than one element is present in\n    ``filters`` list. Either bears or sections need to be passed,\n    if both or none are passed it defaults to use bears gathering\n    and runs filter in bear filtering mode.\n\n    :param filters:\n        OrderedDict of filters based on ``bears`` to be filtered. For example:\n        ``{'language': ('c', 'java'), 'can_fix': ('syntax',),\n        'section_tags': ('save',)}``\n    :param bears:\n        The bears to filter.\n    :param sections:\n        The sections to filter.\n    :return:\n        Filtered bears or sections.\n    "
    items = bears
    applier = apply_filter
    if sections is not None:
        items = sections
        applier = _apply_section_filter
    for (filter_name, filter_args) in filters.items():
        items = applier(filter_name, filter_args, items)
    return items

def _apply_section_filter(filter_name, filter_args, all_sections):
    if False:
        return 10
    "\n    Returns sections after filtering based on ``filter_args``. It\n    returns all sections if nothing is present in ``filter_args``.\n\n    :param filter_name:\n        Name of the section filter.\n    :param filter_args:\n        Arguments to be passed to the filter. For example:\n        ``{'section_tags': ('save', 'change')}``\n    :param all_sections:\n        List of all sections on which filter is to be applied.\n    :return:\n        Filtered sections based on a single section filter.\n    "
    if not is_valid_filter(filter_name):
        raise InvalidFilterException(filter_name)
    if not filter_args or len(filter_args) == 0:
        return all_sections
    filter_function = available_filters[filter_name]
    filtered_sections = []
    for section in all_sections:
        if filter_function(section, filter_args):
            filtered_sections += [section]
    return filtered_sections

def collect_filters(args, arg_list=None, arg_parser=None):
    if False:
        while True:
            i = 10
    "\n    Collects all filters from based on cli arguments.\n\n    :param args:\n        Parsed CLI args using which the filters are to be collected.\n    :param arg_list:\n        The CLI argument list.\n    :param arg_parser:\n        Instance of ArgParser that is used to parse arg list.\n    :return:\n        List of filters in standard filter format, i.e\n        ``{'filter_name': ('arg1', 'arg2')}``.\n    "
    if args is None:
        arg_parser = default_arg_parser() if arg_parser is None else arg_parser
        args = arg_parser.parse_args(arg_list)
    filters = getattr(args, 'filter_by', None) or []
    filters = filter_vector_to_dict(filters)
    return filters

def filter_vector_to_dict(filters):
    if False:
        i = 10
        return i + 15
    "\n    Changes filter vector to OrderedDict.\n\n    :param filters:\n        List of filters in standard filter format, i.e\n        ``[['filter_name', 'arg1', 'arg2']]``.\n    :return:\n        OrderedDict of filters, For example:\n        ``{'filter_name': ('arg1', 'arg2')}``\n    "
    items = OrderedDict()
    for filter_vector in filters:
        (filter_name, args) = (filter_vector[0], tuple(filter_vector[1:]))
        items[filter_name] = args
    return items