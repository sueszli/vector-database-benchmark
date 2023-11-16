import os
import re
import sys
from core.enums import BACKGROUND
from core.enums import COLOR
from core.enums import SEVERITY
IS_TTY = hasattr(sys.stdout, 'fileno') and os.isatty(sys.stdout.fileno())

class ColorizedStream:

    def __init__(self, original):
        if False:
            while True:
                i = 10
        self._original = original
        self._log_colors = {'i': COLOR.LIGHT_BLUE, '!': COLOR.LIGHT_YELLOW, '*': COLOR.LIGHT_CYAN, 'x': COLOR.BOLD_LIGHT_RED, '?': COLOR.LIGHT_YELLOW, 'o': COLOR.BOLD_WHITE, '+': COLOR.BOLD_LIGHT_GREEN, '^': COLOR.BOLD_LIGHT_GREEN}
        self._severity_colors = {SEVERITY.LOW: COLOR.BOLD_LIGHT_CYAN, SEVERITY.MEDIUM: COLOR.BOLD_LIGHT_YELLOW, SEVERITY.HIGH: COLOR.BOLD_LIGHT_RED}
        self._type_colors = {'DNS': BACKGROUND.BLUE, 'UA': BACKGROUND.MAGENTA, 'IP': BACKGROUND.RED, 'URL': BACKGROUND.YELLOW, 'HTTP': BACKGROUND.GREEN, 'IPORT': BACKGROUND.RED}
        self._info_colors = {'malware': COLOR.LIGHT_RED, 'suspicious': COLOR.LIGHT_YELLOW, 'malicious': COLOR.YELLOW}

    def write(self, text):
        if False:
            i = 10
            return i + 15
        match = re.search('\\A(\\s*)\\[(.)\\]', text)
        if match and match.group(2) in self._log_colors:
            text = text.replace(match.group(0), '%s[%s%s%s]' % (match.group(1), self._log_colors[match.group(2)], match.group(2), COLOR.RESET), 1)
        if 'Maltrail (' in text:
            text = re.sub('\\((sensor|server)\\)', lambda match: '(%s%s%s)' % ({'sensor': COLOR.BOLD_LIGHT_GREEN, 'server': COLOR.BOLD_LIGHT_MAGENTA}[match.group(1)], match.group(1), COLOR.RESET), text)
            text = re.sub('https?://[\\w.:/?=]+', lambda match: '%s%s%s%s' % (COLOR.BLUE, COLOR.UNDERLINE, match.group(0), COLOR.RESET), text)
        if 'Usage: ' in text:
            text = re.sub('(.*Usage: )(.+)', '\\g<1>%s\\g<2>%s' % (COLOR.BOLD_WHITE, COLOR.RESET), text)
        if text.startswith('"2'):
            text = re.sub('(TCP|UDP|ICMP) ([A-Z]+)', lambda match: '%s %s%s%s' % (match.group(1), self._type_colors.get(match.group(2), COLOR.WHITE), match.group(2), COLOR.RESET), text)
            text = re.sub('"([^"]+)"', '"%s\\g<1>%s"' % (COLOR.LIGHT_GRAY, COLOR.RESET), text, count=1)
            text = re.sub('\\((malware|suspicious|malicious)\\)', lambda match: '(%s%s%s)' % (self._info_colors.get(match.group(1), COLOR.WHITE), match.group(1), COLOR.RESET), text)
            text = re.sub('\\(([^)]+)\\)', lambda match: '(%s%s%s)' % (COLOR.LIGHT_GRAY, match.group(1), COLOR.RESET) if match.group(1) not in self._info_colors else match.group(0), text)
        for match in re.finditer("[^\\w]'([^']+)'", text):
            text = text.replace("'%s'" % match.group(1), "'%s%s%s'" % (COLOR.LIGHT_GRAY, match.group(1), COLOR.RESET))
        self._original.write('%s' % text)

    def flush(self):
        if False:
            while True:
                i = 10
        self._original.flush()

def init_output():
    if False:
        while True:
            i = 10
    if IS_TTY:
        sys.stderr = ColorizedStream(sys.stderr)
        sys.stdout = ColorizedStream(sys.stdout)