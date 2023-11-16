from __future__ import absolute_import
import re
from st2common.util.actionalias_matching import normalise_alias_format_string
__all__ = ['generate_helpstring_result']

def generate_helpstring_result(aliases, filter=None, pack=None, limit=0, offset=0):
    if False:
        print('Hello World!')
    '\n    List help strings from a collection of alias objects.\n\n    :param aliases: The list of aliases\n    :type  aliases: ``list`` of :class:`st2common.models.api.action.ActionAliasAPI`\n    :param filter_: A search pattern.\n    :type  filter_: ``string``\n    :param pack: Name of a pack\n    :type  pack: ``string``\n    :param limit: The number of help strings to return in the list.\n    :type  limit: ``integer``\n    :param offset: The offset in the list to start returning help strings.\n    :type  limit: ``integer``\n\n    :return: A list of aliases help strings.\n    :rtype: ``list`` of ``list``\n    '
    matches = []
    count = 0
    if not (isinstance(limit, int) and isinstance(offset, int)):
        raise TypeError('limit or offset argument is not an integer')
    for alias in aliases:
        if not alias.enabled:
            continue
        if pack and pack != alias.pack:
            continue
        for format_ in alias.formats:
            (display, _, _) = normalise_alias_format_string(format_)
            if display:
                if not re.search(filter or '', display, flags=re.IGNORECASE):
                    continue
                if (offset == 0 and limit > 0) and count >= limit:
                    count += 1
                    continue
                elif (offset > 0 and limit == 0) and count < offset:
                    count += 1
                    continue
                elif (offset > 0 and limit > 0) and (count < offset or count >= offset + limit):
                    count += 1
                    continue
                matches.append({'pack': alias.pack, 'display': display, 'description': alias.description})
                count += 1
    return {'available': count, 'helpstrings': matches}