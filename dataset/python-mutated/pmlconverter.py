"""
Convert pml markup to and from html
"""
__license__ = 'GPL v3'
__copyright__ = '2009, John Schember <john@nachtimwald.com>'
__docformat__ = 'restructuredtext en'
import os
import re
import io
from copy import deepcopy
from calibre import my_unichr, prepare_string_for_xml
from calibre.ebooks.metadata.toc import TOC

class PML_HTMLizer:
    STATES = ['i', 'u', 'd', 'b', 'sp', 'sb', 'h1', 'h1c', 'h2', 'h3', 'h4', 'h5', 'h6', 'a', 'ra', 'c', 'r', 's', 'l', 'k', 'FN', 'SB']
    STATES_VALUE_REQ = ['a', 'FN', 'SB']
    STATES_VALUE_REQ_2 = ['ra']
    STATES_CLOSE_VALUE_REQ = ['FN', 'SB']
    STATES_TAGS = {'h1': ('<h1 style="page-break-before: always;">', '</h1>'), 'h1c': ('<h1>', '</h1>'), 'h2': ('<h2>', '</h2>'), 'h3': ('<h3>', '</h3>'), 'h4': ('<h4>', '</h4>'), 'h5': ('<h5>', '</h5>'), 'h6': ('<h6>', '</h6>'), 'sp': ('<sup>', '</sup>'), 'sb': ('<sub>', '</sub>'), 'a': ('<a href="#%s">', '</a>'), 'ra': ('<span id="r%s"></span><a href="#%s">', '</a>'), 'c': ('<div style="text-align: center; margin: auto;">', '</div>'), 'r': ('<div style="text-align: right;">', '</div>'), 't': ('<div style="margin-left: 5%;">', '</div>'), 'T': ('<div style="text-indent: %s;">', '</div>'), 'i': ('<span style="font-style: italic;">', '</span>'), 'u': ('<span style="text-decoration: underline;">', '</span>'), 'd': ('<span style="text-decoration: line-through;">', '</span>'), 'b': ('<span style="font-weight: bold;">', '</span>'), 'l': ('<span style="font-size: 150%;">', '</span>'), 'k': ('<span style="font-size: 75%; font-variant: small-caps;">', '</span>'), 'FN': ('<br /><br style="page-break-after: always;" /><div id="fn-%s"><p>', '</p><small><a href="#rfn-%s">return</a></small></div>'), 'SB': ('<br /><br style="page-break-after: always;" /><div id="sb-%s"><p>', '</p><small><a href="#rsb-%s">return</a></small></div>')}
    CODE_STATES = {'q': 'a', 'x': 'h1', 'X0': 'h2', 'X1': 'h3', 'X2': 'h4', 'X3': 'h5', 'X4': 'h6', 'Sp': 'sp', 'Sb': 'sb', 'c': 'c', 'r': 'r', 'i': 'i', 'I': 'i', 'u': 'u', 'o': 'd', 'b': 'b', 'B': 'b', 'l': 'l', 'k': 'k', 'Fn': 'ra', 'Sd': 'ra', 'FN': 'FN', 'SB': 'SB'}
    LINK_STATES = ['a', 'ra']
    BLOCK_STATES = ['a', 'ra', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'sb', 'sp']
    DIV_STATES = ['c', 'r', 'FN', 'SB']
    SPAN_STATES = ['l', 'k', 'i', 'u', 'd', 'b']
    NEW_LINE_EXCHANGE_STATES = {'h1': 'h1c'}

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.state = {}
        self.toc = []
        self.file_name = ''

    def prepare_pml(self, pml):
        if False:
            print('Hello World!')
        pml = re.sub('(?msu)(?P<c>\\\\x)(?P<text>.*?)(?P=c)', lambda match: '%s="%s"%s%s' % (match.group('c'), self.strip_pml(match.group('text')), match.group('text'), match.group('c')), pml)
        pml = re.sub('(?msu)(?P<c>\\\\X[0-4])(?P<text>.*?)(?P=c)', lambda match: '%s="%s"%s%s' % (match.group('c'), self.strip_pml(match.group('text')), match.group('text'), match.group('c')), pml)
        pml = re.sub('(?mus)\\\\v(?P<text>.*?)\\\\v', '', pml)
        pml = re.sub('(?mus)[ ]{2,}', ' ', pml)
        pml = re.sub('(?mus)^[ ]*(?=.)', '', pml)
        pml = re.sub('(?mus)(?<=.)[ ]*$', '', pml)
        pml = re.sub('(?mus)^[ ]*$', '', pml)
        pml = re.sub('(?mus)<footnote\\s+id="(?P<target>.+?)">\\s*(?P<text>.*?)\\s*</footnote>', lambda match: '\\FN="%s"%s\\FN' % (match.group('target'), match.group('text')) if match.group('text') else '', pml)
        pml = re.sub('(?mus)<sidebar\\s+id="(?P<target>.+?)">\\s*(?P<text>.*?)\\s*</sidebar>', lambda match: '\\SB="%s"%s\\SB' % (match.group('target'), match.group('text')) if match.group('text') else '', pml)
        pml = pml.replace('&', '&amp;')
        pml = re.sub('\\\\a(?P<num>\\d{3})', lambda match: '&#%s;' % match.group('num'), pml)
        pml = re.sub('\\\\U(?P<num>[0-9a-f]{4})', lambda match: '%s' % my_unichr(int(match.group('num'), 16)), pml)
        pml = prepare_string_for_xml(pml)
        return pml

    def strip_pml(self, pml):
        if False:
            i = 10
            return i + 15
        pml = re.sub('\\\\C\\d=".*"', '', pml)
        pml = re.sub('\\\\Fn=".*"', '', pml)
        pml = re.sub('\\\\Sd=".*"', '', pml)
        pml = re.sub('\\\\.=".*"', '', pml)
        pml = re.sub('\\\\X\\d', '', pml)
        pml = re.sub('\\\\S[pbd]', '', pml)
        pml = re.sub('\\\\Fn', '', pml)
        pml = re.sub('\\\\a\\d\\d\\d', '', pml)
        pml = re.sub('\\\\U\\d\\d\\d\\d', '', pml)
        pml = re.sub('\\\\.', '', pml)
        pml = pml.replace('\r\n', ' ')
        pml = pml.replace('\n', ' ')
        pml = pml.replace('\r', ' ')
        pml = pml.strip()
        return pml

    def cleanup_html(self, html):
        if False:
            i = 10
            return i + 15
        old = html
        html = self.cleanup_html_remove_redundant(html)
        while html != old:
            old = html
            html = self.cleanup_html_remove_redundant(html)
        html = re.sub('(?imu)^\\s*', '', html)
        return html

    def cleanup_html_remove_redundant(self, html):
        if False:
            for i in range(10):
                print('nop')
        for key in self.STATES_TAGS:
            (open, close) = self.STATES_TAGS[key]
            if key in self.STATES_VALUE_REQ:
                html = re.sub('(?u){}\\s*{}'.format(open % '.*?', close), '', html)
            else:
                html = re.sub(f'(?u){open}\\s*{close}', '', html)
        html = re.sub('(?imu)<p>\\s*</p>', '', html)
        return html

    def start_line(self):
        if False:
            i = 10
            return i + 15
        start = ''
        state = deepcopy(self.state)
        div = []
        span = []
        other = []
        for (key, val) in state.items():
            if key in self.NEW_LINE_EXCHANGE_STATES and val[0]:
                state[self.NEW_LINE_EXCHANGE_STATES[key]] = val
                state[key] = [False, '']
        for (key, val) in state.items():
            if val[0]:
                if key in self.DIV_STATES:
                    div.append((key, val[1]))
                elif key in self.SPAN_STATES:
                    span.append((key, val[1]))
                else:
                    other.append((key, val[1]))
        for (key, val) in other + div + span:
            if key in self.STATES_VALUE_REQ:
                start += self.STATES_TAGS[key][0] % val
            elif key in self.STATES_VALUE_REQ_2:
                start += self.STATES_TAGS[key][0] % (val, val)
            else:
                start += self.STATES_TAGS[key][0]
        return '<p>%s' % start

    def end_line(self):
        if False:
            i = 10
            return i + 15
        end = ''
        div = []
        span = []
        other = []
        for (key, val) in self.state.items():
            if val[0]:
                if key in self.DIV_STATES:
                    div.append(key)
                elif key in self.SPAN_STATES:
                    span.append(key)
                else:
                    other.append(key)
        for key in span + div + other:
            if key in self.STATES_CLOSE_VALUE_REQ:
                end += self.STATES_TAGS[key][1] % self.state[key][1]
            else:
                end += self.STATES_TAGS[key][1]
        return '%s</p>' % end

    def process_code(self, code, stream, pre=''):
        if False:
            while True:
                i = 10
        text = ''
        code = self.CODE_STATES.get(code, None)
        if not code:
            return text
        if code in self.DIV_STATES:
            if code == 'T' and self.state['T'][0]:
                self.code_value(stream)
                return text
            text = self.process_code_div(code, stream)
        elif code in self.SPAN_STATES:
            text = self.process_code_span(code, stream)
        elif code in self.BLOCK_STATES:
            text = self.process_code_block(code, stream, pre)
        else:
            text = self.process_code_simple(code, stream)
        self.state[code][0] = not self.state[code][0]
        return text

    def process_code_simple(self, code, stream):
        if False:
            print('Hello World!')
        text = ''
        if self.state[code][0]:
            if code in self.STATES_CLOSE_VALUE_REQ:
                text = self.STATES_TAGS[code][1] % self.state[code][1]
            else:
                text = self.STATES_TAGS[code][1]
        elif code in self.STATES_VALUE_REQ or code in self.STATES_VALUE_REQ_2:
            val = self.code_value(stream)
            if code in self.STATES_VALUE_REQ:
                text = self.STATES_TAGS[code][0] % val
            else:
                text = self.STATES_TAGS[code][0] % (val, val)
            self.state[code][1] = val
        else:
            text = self.STATES_TAGS[code][0]
        return text

    def process_code_div(self, code, stream):
        if False:
            print('Hello World!')
        text = ''
        if self.state[code][0]:
            for c in self.SPAN_STATES + self.DIV_STATES:
                if self.state[c][0]:
                    if c in self.STATES_CLOSE_VALUE_REQ:
                        text += self.STATES_TAGS[c][1] % self.state[c][1]
                    else:
                        text += self.STATES_TAGS[c][1]
            for c in self.DIV_STATES + self.SPAN_STATES:
                if code == c:
                    continue
                if self.state[c][0]:
                    if c in self.STATES_VALUE_REQ:
                        text += self.STATES_TAGS[self.CODE_STATES[c]][0] % self.state[c][1]
                    elif c in self.STATES_VALUE_REQ_2:
                        text += self.STATES_TAGS[self.CODE_STATES[c]][0] % (self.state[c][1], self.state[c][1])
                    else:
                        text += self.STATES_TAGS[c][0]
        else:
            for c in self.SPAN_STATES:
                if self.state[c][0]:
                    if c in self.STATES_CLOSE_VALUE_REQ:
                        text += self.STATES_TAGS[c][1] % self.state[c][1]
                    else:
                        text += self.STATES_TAGS[c][1]
            if code in self.STATES_VALUE_REQ or code in self.STATES_VALUE_REQ_2:
                val = self.code_value(stream)
                if code in self.STATES_VALUE_REQ:
                    text += self.STATES_TAGS[code][0] % val
                else:
                    text += self.STATES_TAGS[code][0] % (val, val)
                self.state[code][1] = val
            else:
                text += self.STATES_TAGS[code][0]
            for c in self.SPAN_STATES:
                if self.state[c][0]:
                    if c in self.STATES_VALUE_REQ:
                        text += self.STATES_TAGS[self.CODE_STATES[c]][0] % self.state[c][1]
                    elif c in self.STATES_VALUE_REQ_2:
                        text += self.STATES_TAGS[self.CODE_STATES[c]][0] % (self.state[c][1], self.state[c][1])
                    else:
                        text += self.STATES_TAGS[c][0]
        return text

    def process_code_span(self, code, stream):
        if False:
            return 10
        text = ''
        if self.state[code][0]:
            for c in self.SPAN_STATES:
                if self.state[c][0]:
                    if c in self.STATES_CLOSE_VALUE_REQ:
                        text += self.STATES_TAGS[c][1] % self.state[c][1]
                    else:
                        text += self.STATES_TAGS[c][1]
            for c in self.SPAN_STATES:
                if code == c:
                    continue
                if self.state[c][0]:
                    if c in self.STATES_VALUE_REQ:
                        text += self.STATES_TAGS[code][0] % self.state[c][1]
                    elif c in self.STATES_VALUE_REQ_2:
                        text += self.STATES_TAGS[code][0] % (self.state[c][1], self.state[c][1])
                    else:
                        text += self.STATES_TAGS[c][0]
        elif code in self.STATES_VALUE_REQ or code in self.STATES_VALUE_REQ_2:
            val = self.code_value(stream)
            if code in self.STATES_VALUE_REQ:
                text += self.STATES_TAGS[code][0] % val
            else:
                text += self.STATES_TAGS[code][0] % (val, val)
            self.state[code][1] = val
        else:
            text += self.STATES_TAGS[code][0]
        return text

    def process_code_block(self, code, stream, pre=''):
        if False:
            for i in range(10):
                print('nop')
        text = ''
        for c in self.SPAN_STATES:
            if self.state[c][0]:
                if c in self.STATES_CLOSE_VALUE_REQ:
                    text += self.STATES_TAGS[c][1] % self.state[c][1]
                else:
                    text += self.STATES_TAGS[c][1]
        if self.state[code][0]:
            if code in self.STATES_CLOSE_VALUE_REQ:
                text += self.STATES_TAGS[code][1] % self.state[code][1]
            else:
                text += self.STATES_TAGS[code][1]
        elif code in self.STATES_VALUE_REQ or code in self.STATES_VALUE_REQ_2:
            val = self.code_value(stream)
            if code in self.LINK_STATES:
                val = val.lstrip('#')
            if pre:
                val = f'{pre}-{val}'
            if code in self.STATES_VALUE_REQ:
                text += self.STATES_TAGS[code][0] % val
            else:
                text += self.STATES_TAGS[code][0] % (val, val)
            self.state[code][1] = val
        else:
            text += self.STATES_TAGS[code][0]
        for c in self.SPAN_STATES:
            if self.state[c][0]:
                if c in self.STATES_VALUE_REQ:
                    text += self.STATES_TAGS[code][0] % self.state[c][1]
                elif c in self.STATES_VALUE_REQ_2:
                    text += self.STATES_TAGS[code][0] % (self.state[c][1], self.state[c][1])
                else:
                    text += self.STATES_TAGS[c][0]
        return text

    def code_value(self, stream):
        if False:
            return 10
        value = ''
        state = 0
        loc = stream.tell()
        c = stream.read(1)
        while c != '':
            if state == 0:
                if c == '=':
                    state = 1
                elif c != ' ':
                    break
            elif state == 1:
                if c == '"':
                    state = 2
                elif c != ' ':
                    break
            elif state == 2:
                if c == '"':
                    state = 3
                    break
                else:
                    value += c
            c = stream.read(1)
        if state != 3:
            stream.seek(loc)
            value = ''
        return value.strip()

    def parse_pml(self, pml, file_name=''):
        if False:
            for i in range(10):
                print('nop')
        pml = self.prepare_pml(pml)
        output = []
        self.state = {}
        self.toc = []
        self.file_name = file_name
        indent_state = {'t': False, 'T': False, 'st': False, 'sT': False, 'et': False}
        basic_indent = False
        adv_indent_val = ''
        empty_count = 0
        for s in self.STATES:
            self.state[s] = [False, '']
        for line in pml.splitlines():
            parsed = []
            empty = True
            basic_indent = indent_state['t']
            indent_state['T'] = False
            if line.lstrip().startswith('\\t') or basic_indent:
                basic_indent = True
                indent_state['st'] = True
            else:
                indent_state['st'] = False
            if line.lstrip().startswith('\\T'):
                indent_state['sT'] = True
            else:
                indent_state['sT'] = False
            if line.rstrip().endswith('\\t'):
                indent_state['et'] = True
            else:
                indent_state['et'] = False
            if isinstance(line, bytes):
                line = line.decode('utf-8')
            line = io.StringIO(line)
            parsed.append(self.start_line())
            c = line.read(1)
            while c != '':
                text = ''
                if c == '\\':
                    c = line.read(1)
                    if c in 'qcriIuobBlk':
                        text = self.process_code(c, line)
                    elif c in 'FS':
                        l = line.read(1)
                        if f'{c}{l}' == 'Fn':
                            text = self.process_code('Fn', line, 'fn')
                        elif f'{c}{l}' == 'FN':
                            text = self.process_code('FN', line)
                        elif f'{c}{l}' == 'SB':
                            text = self.process_code('SB', line)
                        elif f'{c}{l}' == 'Sd':
                            text = self.process_code('Sd', line, 'sb')
                    elif c in 'xXC':
                        empty = False
                        t = ''
                        level = 0
                        if c in 'XC':
                            level = line.read(1)
                        id = 'pml_toc-%s' % len(self.toc)
                        value = self.code_value(line)
                        if c == 'x':
                            t = self.process_code(c, line)
                        elif c == 'X':
                            t = self.process_code(f'{c}{level}', line)
                        if not value or value == '':
                            text = t
                        else:
                            self.toc.append((level, (os.path.basename(self.file_name), id, value)))
                            text = f'{t}<span id="{id}"></span>'
                    elif c == 'm':
                        empty = False
                        src = self.code_value(line)
                        text = '<img src="images/%s" />' % src
                    elif c == 'Q':
                        empty = False
                        id = self.code_value(line)
                        text = '<span id="%s"></span>' % id
                    elif c == 'p':
                        empty = False
                        text = '<br /><br style="page-break-after: always;" />'
                    elif c == 'n':
                        pass
                    elif c == 'w':
                        empty = False
                        text = '<hr style="width: %s" />' % self.code_value(line)
                    elif c == 't':
                        indent_state['t'] = not indent_state['t']
                    elif c == 'T':
                        if not indent_state['T']:
                            adv_indent_val = self.code_value(line)
                        else:
                            self.code_value(line)
                        indent_state['T'] = True
                    elif c == '-':
                        empty = False
                        text = '&shy;'
                    elif c == '\\':
                        empty = False
                        text = '\\'
                else:
                    if c != ' ':
                        empty = False
                    text = c
                parsed.append(text)
                c = line.read(1)
            if empty:
                empty_count += 1
                if empty_count == 2:
                    output.append('<p>&nbsp;</p>')
            else:
                empty_count = 0
                text = self.end_line()
                parsed.append(text)
                if basic_indent:
                    if indent_state['st'] and (indent_state['et'] or indent_state['t']):
                        parsed.insert(0, self.STATES_TAGS['t'][0])
                        parsed.append(self.STATES_TAGS['t'][1])
                    else:
                        parsed.insert(0, self.STATES_TAGS['T'][0] % '5%')
                        parsed.append(self.STATES_TAGS['T'][1])
                elif indent_state['T'] and indent_state['sT']:
                    parsed.insert(0, self.STATES_TAGS['T'][0] % adv_indent_val)
                    parsed.append(self.STATES_TAGS['T'][1])
                    indent_state['T'] = False
                    adv_indent_val = ''
                output.append(''.join(parsed))
            line.close()
        output = self.cleanup_html('\n'.join(output))
        return output

    def get_toc(self):
        if False:
            return 10
        '\n        Toc can have up to 5 levels, 0 - 4 inclusive.\n\n        This function will add items to their appropriate\n        depth in the TOC tree. If the specified depth is\n        invalid (item would not have a valid parent) add\n        it to the next valid level above the specified\n        level.\n        '
        n_toc = TOC()
        t_l0 = None
        t_l1 = None
        t_l2 = None
        t_l3 = None
        for (level, (href, id, text)) in self.toc:
            if level == '0':
                t_l0 = n_toc.add_item(href, id, text)
                t_l1 = None
                t_l2 = None
                t_l3 = None
            elif level == '1':
                if t_l0 is None:
                    t_l0 = n_toc
                t_l1 = t_l0.add_item(href, id, text)
                t_l2 = None
                t_l3 = None
            elif level == '2':
                if t_l1 is None:
                    if t_l0 is None:
                        t_l1 = n_toc
                    else:
                        t_l1 = t_l0
                t_l2 = t_l1.add_item(href, id, text)
                t_l3 = None
            elif level == '3':
                if t_l2 is None:
                    if t_l1 is None:
                        if t_l0 is None:
                            t_l2 = n_toc
                        else:
                            t_l2 = t_l0
                    else:
                        t_l2 = t_l1
                t_l3 = t_l2.add_item(href, id, text)
            else:
                if t_l3 is None:
                    if t_l2 is None:
                        if t_l1 is None:
                            if t_l0 is None:
                                t_l3 = n_toc
                            else:
                                t_l3 = t_l0
                        else:
                            t_l3 = t_l1
                    else:
                        t_l3 = t_l2
                t_l3.add_item(href, id, text)
        return n_toc

def pml_to_html(pml):
    if False:
        for i in range(10):
            print('nop')
    hizer = PML_HTMLizer()
    return hizer.parse_pml(pml)

def footnote_sidebar_to_html(pre_id, id, pml):
    if False:
        return 10
    id = id.strip('\x01')
    if id.strip():
        html = '<br /><br style="page-break-after: always;" /><div id="{}-{}">{}<small><a href="#r{}-{}">return</a></small></div>'.format(pre_id, id, pml_to_html(pml), pre_id, id)
    else:
        html = '<br /><br style="page-break-after: always;" /><div>%s</div>' % pml_to_html(pml)
    return html

def footnote_to_html(id, pml):
    if False:
        for i in range(10):
            print('nop')
    return footnote_sidebar_to_html('fn', id, pml)

def sidebar_to_html(id, pml):
    if False:
        for i in range(10):
            print('nop')
    return footnote_sidebar_to_html('sb', id, pml)