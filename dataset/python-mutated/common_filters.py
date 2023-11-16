import re
__all__ = ['URL_RE', 'INVITE_URL_RE', 'MASS_MENTION_RE', 'filter_urls', 'filter_invites', 'filter_mass_mentions', 'filter_various_mentions', 'normalize_smartquotes', 'escape_spoilers', 'escape_spoilers_and_mass_mentions']
URL_RE = re.compile('(https?|s?ftp)://(\\S+)', re.I)
INVITE_URL_RE = re.compile('(discord\\.(?:gg|io|me|li)|discord(?:app)?\\.com\\/invite)\\/(\\S+)', re.I)
MASS_MENTION_RE = re.compile('(@)(?=everyone|here)')
OTHER_MENTION_RE = re.compile('(<)(@[!&]?|#)(\\d+>)')
SMART_QUOTE_REPLACEMENT_DICT = {'‘': "'", '’': "'", '“': '"', '”': '"'}
SMART_QUOTE_REPLACE_RE = re.compile('|'.join(SMART_QUOTE_REPLACEMENT_DICT.keys()))
SPOILER_CONTENT_RE = re.compile('(?s)(?<!\\\\)(?P<OPEN>\\|{2})(?P<SPOILERED>.*?)(?<!\\\\)(?P<CLOSE>\\|{2})')

def filter_urls(to_filter: str) -> str:
    if False:
        return 10
    'Get a string with URLs sanitized.\n\n    This will match any URLs starting with these protocols:\n\n     - ``http://``\n     - ``https://``\n     - ``ftp://``\n     - ``sftp://``\n\n    Parameters\n    ----------\n    to_filter : str\n        The string to filter.\n\n    Returns\n    -------\n    str\n        The sanitized string.\n\n    '
    return URL_RE.sub('[SANITIZED URL]', to_filter)

def filter_invites(to_filter: str) -> str:
    if False:
        for i in range(10):
            print('nop')
    'Get a string with discord invites sanitized.\n\n    Will match any discord.gg, discordapp.com/invite, discord.com/invite, discord.me, or discord.io/discord.li\n    invite URL.\n\n    Parameters\n    ----------\n    to_filter : str\n        The string to filter.\n\n    Returns\n    -------\n    str\n        The sanitized string.\n\n    '
    return INVITE_URL_RE.sub('[SANITIZED INVITE]', to_filter)

def filter_mass_mentions(to_filter: str) -> str:
    if False:
        while True:
            i = 10
    'Get a string with mass mentions sanitized.\n\n    Will match any *here* and/or *everyone* mentions.\n\n    Parameters\n    ----------\n    to_filter : str\n        The string to filter.\n\n    Returns\n    -------\n    str\n        The sanitized string.\n\n    '
    return MASS_MENTION_RE.sub('@\u200b', to_filter)

def filter_various_mentions(to_filter: str) -> str:
    if False:
        return 10
    '\n    Get a string with role, user, and channel mentions sanitized.\n\n    This is mainly for use on user display names, not message content,\n    and should be applied sparingly.\n\n    Parameters\n    ----------\n    to_filter : str\n        The string to filter.\n\n    Returns\n    -------\n    str\n        The sanitized string.\n    '
    return OTHER_MENTION_RE.sub('\\1\\\\\\2\\3', to_filter)

def normalize_smartquotes(to_normalize: str) -> str:
    if False:
        for i in range(10):
            print('nop')
    '\n    Get a string with smart quotes replaced with normal ones\n\n    Parameters\n    ----------\n    to_normalize : str\n        The string to normalize.\n\n    Returns\n    -------\n    str\n        The normalized string.\n    '

    def replacement_for(obj):
        if False:
            i = 10
            return i + 15
        return SMART_QUOTE_REPLACEMENT_DICT.get(obj.group(0), '')
    return SMART_QUOTE_REPLACE_RE.sub(replacement_for, to_normalize)

def escape_spoilers(content: str) -> str:
    if False:
        return 10
    '\n    Get a string with spoiler syntax escaped.\n\n    Parameters\n    ----------\n    content : str\n        The string to escape.\n\n    Returns\n    -------\n    str\n        The escaped string.\n    '
    return SPOILER_CONTENT_RE.sub('\\\\\\g<OPEN>\\g<SPOILERED>\\\\\\g<CLOSE>', content)

def escape_spoilers_and_mass_mentions(content: str) -> str:
    if False:
        return 10
    '\n    Get a string with spoiler syntax and mass mentions escaped\n\n    Parameters\n    ----------\n    content : str\n        The string to escape.\n\n    Returns\n    -------\n    str\n        The escaped string.\n    '
    return escape_spoilers(filter_mass_mentions(content))