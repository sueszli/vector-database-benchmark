import re
LANGUAGES = {'Afrikaans': 'af', 'Albanian': 'sq', 'Arabic': 'ar', 'Asturian': 'ast', 'Basque': 'eu', 'Belarusian': 'be', 'Bosnian': 'bs', 'Breton': 'br', 'Bulgarian': 'bg', 'Catalan': 'ca', 'Cibemba': 'bem', 'Corsican': 'co', 'Croation': 'hr', 'Czech': 'cs', 'Danish': 'da', 'Dutch': 'nl', 'English': 'en', 'Esperanto': 'eo', 'Estonian': 'et', 'Farsi': 'fa', 'Finnish': 'fi', 'French': 'fr', 'Galician': 'gl', 'Georgian': 'ka', 'German': 'de', 'Greek': 'el', 'Hebrew': 'he', 'Hindi': 'hi', 'Hungarian': 'hu', 'Icelandic': 'is', 'Igbo': 'ig', 'Indonesian': 'id', 'Irish': 'ga', 'Italian': 'it', 'Japanese': 'ja', 'Khmer': 'km', 'Korean': 'ko', 'Kurdish': 'ku', 'Latvian': 'lv', 'Lithuanian': 'lt', 'Luxembourgish': 'lb', 'Macedonian': 'mk', 'Malagasy': 'mg', 'Malay': 'ms_MY', 'Mongolian': 'mn', 'Norwegian': 'nb', 'NorwegianNynorsk': 'nn', 'Polish': 'pl', 'Portuguese': 'pt', 'PortugueseBR': 'pt_BR', 'Romanian': 'ro', 'Russian': 'ru', 'ScotsGaelic': 'sco', 'Serbian': 'sr', 'SimpChinese': 'zh_CN', 'Slovak': 'sk', 'Slovenian': 'sl', 'Spanish': 'es', 'Swahili': 'sw', 'Swedish': 'sv', 'Tatar': 'tt', 'Thai': 'th', 'TradChinese': 'zh_TW', 'Turkish': 'tr', 'Ukrainian': 'uk', 'Uzbek': 'uz', 'Vietnamese': 'vi', 'Welsh': 'cy', 'Yoruba': 'yo'}
_R_LANGUAGES = {code: name for (name, code) in LANGUAGES.items()}
ESCAPE_CHARS = {'$\\r': '\r', '$\\n': '\n', '$\\t': '\t', '$\\"': '"', "$\\'": "'", '$\\`': '`'}
RE_LANGSTRING_LINE = re.compile('LangString\\s+(?P<identifier>[A-Za-z0-9_]+)\\s+\\${LANG_[A-Z]+}\\s+["\\\'`](?P<text>.*)["\\\'`]$')

def language_to_code(language):
    if False:
        i = 10
        return i + 15
    return LANGUAGES.get(language)

def code_to_language(language_code):
    if False:
        for i in range(10):
            print('nop')
    return _R_LANGUAGES.get(language_code)

def escape_string(text):
    if False:
        while True:
            i = 10
    for (escape, char) in ESCAPE_CHARS.items():
        if char not in {"'", '`'}:
            text = text.replace(char, escape)
    return text

def unescape_string(text):
    if False:
        i = 10
        return i + 15
    for (escape, char) in ESCAPE_CHARS.items():
        text = text.replace(escape, char)
    return text

def parse_langstring(line):
    if False:
        print('Hello World!')
    match = RE_LANGSTRING_LINE.match(line)
    if match:
        return (match.group('identifier'), unescape_string(match.group('text')))
    else:
        return None

def make_langstring(language, identifier, text):
    if False:
        return 10
    language = language.upper()
    text = escape_string(text)
    return f'LangString {identifier} ${{LANG_{language}}} "{text}"\n'