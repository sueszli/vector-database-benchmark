import re
from typing import List, Match, Tuple
from bs4 import BeautifulSoup
IGNORED_PHRASES = ['API', 'APNS', 'Botserver', 'Cookie Bot', 'DevAuthBackend', 'DSN', 'GCM', 'GitHub', 'Gravatar', 'Help Center', 'HTTP', 'ID', 'IDs', 'IP', 'JSON', 'Kerberos', 'LDAP', 'Markdown', 'OTP', 'Pivotal', 'DM', 'DMs', 'Slack', 'Google', 'Terms of Service', 'Tuesday', 'URL', 'UUID', 'Webathena', 'WordPress', 'Zephyr', 'Zoom', 'Zulip', 'Zulip Server', 'Zulip Account Security', 'Zulip Security', 'Zulip Cloud Standard', 'BigBlueButton', '\\.zuliprc', '<z-user></z-user> will have the same role', '<z-user></z-user> will have the same properties', 'I understand', "I'm", "I've", 'Topics I participate in', 'Topics I send a message to', 'Topics I start', 'beta', 'and', 'bot', 'e\\.g\\.', 'enabled', 'signups', 'keyword', 'streamname', 'user@example\\.com', 'acme', 'is …', 'your subscriptions on your Streams page', 'Add global time<br />Everyone sees global times in their own time zone\\.', 'user', 'an unknown operating system', 'Go to Settings', 'more topics', '^deprecated$', 'more conversations', 'back to streams', 'in 1 hour', 'in 20 minutes', 'in 3 hours', '^new streams$', '^stream events$', '^marketing$', '^cookie$', '\\bN\\b', 'clear', 'group direct messages with \\{recipient\\}', 'direct messages with \\{recipient\\}', 'direct messages with yourself', 'GIF', 'leafy green vegetable', 'your-organization-url', 'or', 'rated Y', 'rated G', 'rated PG', 'rated PG13', 'rated R', 'GIFs', 'GIPHY', 'Technical University of Munich', 'University of California San Diego', 'email hidden', 'to send', 'to add a new line', 'Notification Bot', 'invisible mode off', 'he/him', 'she/her', 'they/them', 'does not apply to moderators and administrators', 'does not apply to administrators', 'guest', 'deactivated']
IGNORED_PHRASES.sort(key=lambda regex: len(regex), reverse=True)
COMPILED_IGNORED_PHRASES = [re.compile(' '.join(BeautifulSoup(regex, 'lxml').text.split())) for regex in IGNORED_PHRASES]
SPLIT_BOUNDARY = '?.!'
SPLIT_BOUNDARY_REGEX = re.compile(f'[{SPLIT_BOUNDARY}]')
DISALLOWED = ['^[a-z](?!\\})', '^[A-Z][a-z]+[\\sa-z0-9]+[A-Z]']
DISALLOWED_REGEX = re.compile('|'.join(DISALLOWED))
BANNED_WORDS = {'realm': 'The term realm should not appear in user-facing strings. Use organization instead.'}

def get_safe_phrase(phrase: str) -> str:
    if False:
        print('Hello World!')
    "\n    Safe phrase is in lower case and doesn't contain characters which can\n    conflict with split boundaries. All conflicting characters are replaced\n    with low dash (_).\n    "
    phrase = SPLIT_BOUNDARY_REGEX.sub('_', phrase)
    return phrase.lower()

def replace_with_safe_phrase(matchobj: Match[str]) -> str:
    if False:
        i = 10
        return i + 15
    '\n    The idea is to convert IGNORED_PHRASES into safe phrases, see\n    `get_safe_phrase()` function. The only exception is when the\n    IGNORED_PHRASE is at the start of the text or after a split\n    boundary; in this case, we change the first letter of the phrase\n    to upper case.\n    '
    ignored_phrase = matchobj.group(0)
    safe_string = get_safe_phrase(ignored_phrase)
    start_index = matchobj.start()
    complete_string = matchobj.string
    is_string_start = start_index == 0
    punctuation = complete_string[max(start_index - 2, 0)]
    is_after_split_boundary = punctuation in SPLIT_BOUNDARY
    if is_string_start or is_after_split_boundary:
        return safe_string.capitalize()
    return safe_string

def get_safe_text(text: str) -> str:
    if False:
        i = 10
        return i + 15
    '\n    This returns text which is rendered by BeautifulSoup and is in the\n    form that can be split easily and has all IGNORED_PHRASES processed.\n    '
    soup = BeautifulSoup(text, 'lxml')
    text = ' '.join(soup.text.split())
    for phrase_regex in COMPILED_IGNORED_PHRASES:
        text = phrase_regex.sub(replace_with_safe_phrase, text)
    return text

def is_capitalized(safe_text: str) -> bool:
    if False:
        while True:
            i = 10
    sentences = SPLIT_BOUNDARY_REGEX.split(safe_text)
    return not any((DISALLOWED_REGEX.search(sentence.strip()) for sentence in sentences))

def check_banned_words(text: str) -> List[str]:
    if False:
        while True:
            i = 10
    lower_cased_text = text.lower()
    errors = []
    for (word, reason) in BANNED_WORDS.items():
        if word in lower_cased_text:
            if 'realm_name' in lower_cased_text or 'realm_uri' in lower_cased_text:
                continue
            kwargs = dict(word=word, text=text, reason=reason)
            msg = "{word} found in '{text}'. {reason}".format(**kwargs)
            errors.append(msg)
    return errors

def check_capitalization(strings: List[str]) -> Tuple[List[str], List[str], List[str]]:
    if False:
        i = 10
        return i + 15
    errors = []
    ignored = []
    banned_word_errors = []
    for text in strings:
        text = ' '.join(text.split())
        safe_text = get_safe_text(text)
        has_ignored_phrase = text != safe_text
        capitalized = is_capitalized(safe_text)
        if not capitalized:
            errors.append(text)
        elif has_ignored_phrase:
            ignored.append(text)
        banned_word_errors.extend(check_banned_words(text))
    return (sorted(errors), sorted(ignored), sorted(banned_word_errors))