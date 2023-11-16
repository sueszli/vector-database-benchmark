"""
A Python-Markdown extension to treat newlines as hard breaks; like
GitHub-flavored Markdown does.

See the [documentation](https://Python-Markdown.github.io/extensions/nl2br)
for details.
"""
from __future__ import annotations
from . import Extension
from ..inlinepatterns import SubstituteTagInlineProcessor
BR_RE = '\\n'

class Nl2BrExtension(Extension):

    def extendMarkdown(self, md):
        if False:
            i = 10
            return i + 15
        ' Add a `SubstituteTagInlineProcessor` to Markdown. '
        br_tag = SubstituteTagInlineProcessor(BR_RE, 'br')
        md.inlinePatterns.register(br_tag, 'nl', 5)

def makeExtension(**kwargs):
    if False:
        return 10
    return Nl2BrExtension(**kwargs)