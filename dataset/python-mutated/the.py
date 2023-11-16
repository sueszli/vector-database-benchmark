"""Moves patterns in path formats (suitable for moving articles)."""
import re
from beets.plugins import BeetsPlugin
__author__ = 'baobab@heresiarch.info'
__version__ = '1.1'
PATTERN_THE = '^the\\s'
PATTERN_A = '^[a][n]?\\s'
FORMAT = '{0}, {1}'

class ThePlugin(BeetsPlugin):
    patterns = []

    def __init__(self):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.template_funcs['the'] = self.the_template_func
        self.config.add({'the': True, 'a': True, 'format': '{0}, {1}', 'strip': False, 'patterns': []})
        self.patterns = self.config['patterns'].as_str_seq()
        for p in self.patterns:
            if p:
                try:
                    re.compile(p)
                except re.error:
                    self._log.error('invalid pattern: {0}', p)
                else:
                    if not (p.startswith('^') or p.endswith('$')):
                        self._log.warning('warning: "{0}" will not match string start/end', p)
        if self.config['a']:
            self.patterns = [PATTERN_A] + self.patterns
        if self.config['the']:
            self.patterns = [PATTERN_THE] + self.patterns
        if not self.patterns:
            self._log.warning('no patterns defined!')

    def unthe(self, text, pattern):
        if False:
            while True:
                i = 10
        'Moves pattern in the path format string or strips it\n\n        text -- text to handle\n        pattern -- regexp pattern (case ignore is already on)\n        strip -- if True, pattern will be removed\n        '
        if text:
            r = re.compile(pattern, flags=re.IGNORECASE)
            try:
                t = r.findall(text)[0]
            except IndexError:
                return text
            else:
                r = re.sub(r, '', text).strip()
                if self.config['strip']:
                    return r
                else:
                    fmt = self.config['format'].as_str()
                    return fmt.format(r, t.strip()).strip()
        else:
            return ''

    def the_template_func(self, text):
        if False:
            return 10
        if not self.patterns:
            return text
        if text:
            for p in self.patterns:
                r = self.unthe(text, p)
                if r != text:
                    self._log.debug('"{0}" -> "{1}"', text, r)
                    break
            return r
        else:
            return ''