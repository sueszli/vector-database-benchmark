from coalib.parsing.filters.decorators import typed_filter

@typed_filter(('bearclass', 'Bear', 'Section'))
def section_tags_filter(section_or_bear, args):
    if False:
        for i in range(10):
            print('nop')
    '\n    Filters the bears or sections by ``tags``.\n\n    :param section_or_bear: A section or bear instance on which filtering\n                            needs to be carried out.\n    :param args:            Set of tags on which it needs to be filtered.\n    :return:                ``True`` if this instance matches the criteria\n                            inside args, ``False`` otherwise.\n    '
    enabled_tags = list(map(str.lower, args))
    if len(enabled_tags) == 0:
        return True
    section = section_or_bear
    if hasattr(section_or_bear, 'section'):
        section = section_or_bear.section
    section_tags = section.get('tags', False)
    if str(section_tags) == 'False':
        return False
    section_tags = map(str.lower, section_tags)
    return bool(set(section_tags) & set(enabled_tags))