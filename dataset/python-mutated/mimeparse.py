import cgi
__version__ = '1.6.0'
__author__ = 'Joe Gregorio'
__email__ = 'joe@bitworking.org'
__license__ = 'MIT License'
__credits__ = ''

class MimeTypeParseException(ValueError):
    pass

def parse_mime_type(mime_type):
    if False:
        for i in range(10):
            print('nop')
    "Parses a mime-type into its component parts.\n\n    Carves up a mime-type and returns a tuple of the (type, subtype, params)\n    where 'params' is a dictionary of all the parameters for the media range.\n    For example, the media range 'application/xhtml;q=0.5' would get parsed\n    into:\n\n       ('application', 'xhtml', {'q', '0.5'})\n\n    :rtype: (str,str,dict)\n    "
    (full_type, params) = cgi.parse_header(mime_type)
    if full_type == '*':
        full_type = '*/*'
    type_parts = full_type.split('/') if '/' in full_type else None
    if not type_parts or len(type_parts) > 2:
        raise MimeTypeParseException('Can\'t parse type "{}"'.format(full_type))
    (type, subtype) = type_parts
    return (type.strip(), subtype.strip(), params)

def parse_media_range(range):
    if False:
        print('Hello World!')
    "Parse a media-range into its component parts.\n\n    Carves up a media range and returns a tuple of the (type, subtype,\n    params) where 'params' is a dictionary of all the parameters for the media\n    range.  For example, the media range 'application/*;q=0.5' would get parsed\n    into:\n\n       ('application', '*', {'q', '0.5'})\n\n    In addition this function also guarantees that there is a value for 'q'\n    in the params dictionary, filling it in with a proper default if\n    necessary.\n\n    :rtype: (str,str,dict)\n    "
    (type, subtype, params) = parse_mime_type(range)
    params.setdefault('q', params.pop('Q', None))
    try:
        if not params['q'] or not 0 <= float(params['q']) <= 1:
            params['q'] = '1'
    except ValueError:
        params['q'] = '1'
    return (type, subtype, params)

def quality_and_fitness_parsed(mime_type, parsed_ranges):
    if False:
        print('Hello World!')
    "Find the best match for a mime-type amongst parsed media-ranges.\n\n    Find the best match for a given mime-type against a list of media_ranges\n    that have already been parsed by parse_media_range(). Returns a tuple of\n    the fitness value and the value of the 'q' quality parameter of the best\n    match, or (-1, 0) if no match was found. Just as for quality_parsed(),\n    'parsed_ranges' must be a list of parsed media ranges.\n\n    :rtype: (float,int)\n    "
    best_fitness = -1
    best_fit_q = 0
    (target_type, target_subtype, target_params) = parse_media_range(mime_type)
    for (type, subtype, params) in parsed_ranges:
        type_match = type in (target_type, '*') or target_type == '*'
        subtype_match = subtype in (target_subtype, '*') or target_subtype == '*'
        if type_match and subtype_match:
            fitness = type == target_type and 100 or 0
            fitness += subtype == target_subtype and 10 or 0
            param_matches = sum([1 for (key, value) in target_params.items() if key != 'q' and key in params and (value == params[key])])
            fitness += param_matches
            fitness += float(target_params.get('q', 1))
            if fitness > best_fitness:
                best_fitness = fitness
                best_fit_q = params['q']
    return (float(best_fit_q), best_fitness)

def quality_parsed(mime_type, parsed_ranges):
    if False:
        i = 10
        return i + 15
    "Find the best match for a mime-type amongst parsed media-ranges.\n\n    Find the best match for a given mime-type against a list of media_ranges\n    that have already been parsed by parse_media_range(). Returns the 'q'\n    quality parameter of the best match, 0 if no match was found. This function\n    behaves the same as quality() except that 'parsed_ranges' must be a list of\n    parsed media ranges.\n\n    :rtype: float\n    "
    return quality_and_fitness_parsed(mime_type, parsed_ranges)[0]

def quality(mime_type, ranges):
    if False:
        return 10
    "Return the quality ('q') of a mime-type against a list of media-ranges.\n\n    Returns the quality 'q' of a mime-type when compared against the\n    media-ranges in ranges. For example:\n\n    >>> quality('text/html','text/*;q=0.3, text/html;q=0.7,\n                  text/html;level=1, text/html;level=2;q=0.4, */*;q=0.5')\n    0.7\n\n    :rtype: float\n    "
    parsed_ranges = [parse_media_range(r) for r in ranges.split(',')]
    return quality_parsed(mime_type, parsed_ranges)

def best_match(supported, header):
    if False:
        print('Hello World!')
    "Return mime-type with the highest quality ('q') from list of candidates.\n\n    Takes a list of supported mime-types and finds the best match for all the\n    media-ranges listed in header. The value of header must be a string that\n    conforms to the format of the HTTP Accept: header. The value of 'supported'\n    is a list of mime-types. The list of supported mime-types should be sorted\n    in order of increasing desirability, in case of a situation where there is\n    a tie.\n\n    >>> best_match(['application/xbel+xml', 'text/xml'],\n                   'text/*;q=0.5,*/*; q=0.1')\n    'text/xml'\n\n    :rtype: str\n    "
    split_header = _filter_blank(header.split(','))
    parsed_header = [parse_media_range(r) for r in split_header]
    weighted_matches = []
    pos = 0
    for mime_type in supported:
        weighted_matches.append((quality_and_fitness_parsed(mime_type, parsed_header), pos, mime_type))
        pos += 1
    weighted_matches.sort()
    return weighted_matches[-1][0][0] and weighted_matches[-1][2] or ''

def _filter_blank(i):
    if False:
        for i in range(10):
            print('nop')
    'Return all non-empty items in the list.'
    for s in i:
        if s.strip():
            yield s