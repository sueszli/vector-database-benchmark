import re
from typing import Any, List, Match
from markdown import Markdown
from markdown.extensions import Extension
from markdown.preprocessors import Preprocessor
from typing_extensions import override
from zerver.lib.emoji import EMOTICON_CONVERSIONS, name_to_codepoint
from zerver.lib.markdown.priorities import PREPROCESSOR_PRIORITES
REGEXP = re.compile('\\{emoticon_translations\\}')
TABLE_HTML = '<table>\n    <thead>\n        <tr>\n            <th>Emoticon</th>\n            <th>Emoji</th>\n        </tr>\n    </thead>\n    <tbody>\n        {body}\n    </tbody>\n</table>\n'
ROW_HTML = '<tr>\n    <td><code>{emoticon}</code></td>\n    <td>\n        <img\n            src="/static/generated/emoji/images-google-64/{codepoint}.png"\n            alt="{name}"\n            class="emoji-big">\n    </td>\n</tr>\n'

class EmoticonTranslationsHelpExtension(Extension):

    @override
    def extendMarkdown(self, md: Markdown) -> None:
        if False:
            return 10
        'Add SettingHelpExtension to the Markdown instance.'
        md.registerExtension(self)
        md.preprocessors.register(EmoticonTranslation(), 'emoticon_translations', PREPROCESSOR_PRIORITES['emoticon_translations'])

class EmoticonTranslation(Preprocessor):

    @override
    def run(self, lines: List[str]) -> List[str]:
        if False:
            for i in range(10):
                print('nop')
        for (loc, line) in enumerate(lines):
            match = REGEXP.search(line)
            if match:
                text = self.handleMatch(match)
                lines = lines[:loc] + text + lines[loc + 1:]
                break
        return lines

    def handleMatch(self, match: Match[str]) -> List[str]:
        if False:
            print('Hello World!')
        rows = [ROW_HTML.format(emoticon=emoticon, name=name.strip(':'), codepoint=name_to_codepoint[name.strip(':')]) for (emoticon, name) in EMOTICON_CONVERSIONS.items()]
        body = ''.join(rows).strip()
        return TABLE_HTML.format(body=body).strip().splitlines()

def makeExtension(*args: Any, **kwargs: Any) -> EmoticonTranslationsHelpExtension:
    if False:
        for i in range(10):
            print('nop')
    return EmoticonTranslationsHelpExtension(*args, **kwargs)