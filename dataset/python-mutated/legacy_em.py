"""
This extension provides legacy behavior for _connected_words_.
"""
from __future__ import annotations
from . import Extension
from ..inlinepatterns import UnderscoreProcessor, EmStrongItem, EM_STRONG2_RE, STRONG_EM2_RE
import re
EMPHASIS_RE = '(_)([^_]+)\\1'
STRONG_RE = '(_{2})(.+?)\\1'
STRONG_EM_RE = '(_)\\1(?!\\1)([^_]+?)\\1(?!\\1)(.+?)\\1{3}'

class LegacyUnderscoreProcessor(UnderscoreProcessor):
    """Emphasis processor for handling strong and em matches inside underscores."""
    PATTERNS = [EmStrongItem(re.compile(EM_STRONG2_RE, re.DOTALL | re.UNICODE), 'double', 'strong,em'), EmStrongItem(re.compile(STRONG_EM2_RE, re.DOTALL | re.UNICODE), 'double', 'em,strong'), EmStrongItem(re.compile(STRONG_EM_RE, re.DOTALL | re.UNICODE), 'double2', 'strong,em'), EmStrongItem(re.compile(STRONG_RE, re.DOTALL | re.UNICODE), 'single', 'strong'), EmStrongItem(re.compile(EMPHASIS_RE, re.DOTALL | re.UNICODE), 'single', 'em')]

class LegacyEmExtension(Extension):
    """ Add legacy_em extension to Markdown class."""

    def extendMarkdown(self, md):
        if False:
            while True:
                i = 10
        ' Modify inline patterns. '
        md.inlinePatterns.register(LegacyUnderscoreProcessor('_'), 'em_strong2', 50)

def makeExtension(**kwargs):
    if False:
        print('Hello World!')
    ' Return an instance of the `LegacyEmExtension` '
    return LegacyEmExtension(**kwargs)