import urwid
import re
import sys
import os
from bs4 import BeautifulSoup
import requests
from queue import Queue
from subprocess import PIPE, Popen
from threading import Thread
import webbrowser
import time
from urwid.widget import BOX, FLOW, FIXED
import random
SO_URL = 'https://stackoverflow.com'
GREEN = '\x1b[92m'
GRAY = '\x1b[90m'
CYAN = '\x1b[36m'
RED = '\x1b[31m'
YELLOW = '\x1b[33m'
END = '\x1b[0m'
UNDERLINE = '\x1b[4m'
BOLD = '\x1b[1m'
SCROLL_LINE_UP = 'line up'
SCROLL_LINE_DOWN = 'line down'
SCROLL_PAGE_UP = 'page up'
SCROLL_PAGE_DOWN = 'page down'
SCROLL_TO_TOP = 'to top'
SCROLL_TO_END = 'to end'
SCROLLBAR_LEFT = 'left'
SCROLLBAR_RIGHT = 'right'
USER_AGENTS = ['Mozilla/5.0 (compatible; MSIE 6.0; Windows NT 5.1; SV1; AcooBrowser; .NET CLR 1.1.4322; .NET CLR 2.0.50727)', 'Mozilla/5.0 (Windows; U; MSIE 9.0; Windows NT 9.0; en-US)', 'Mozilla/5.0 (Windows; U; Windows NT 5.1; zh-CN) AppleWebKit/523.15 (KHTML, like Gecko, Safari/419.3) Arora/0.3 (Change: 287 c9dfb30)', 'Mozilla/5.0 (X11; U; Linux; en-US) AppleWebKit/527+ (KHTML, like Gecko, Safari/419.3) Arora/0.6', 'Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US; rv:1.8.1.2pre) Gecko/20070215 K-Ninja/2.1.1', 'Mozilla/5.0 (Windows; U; Windows NT 5.1; zh-CN; rv:1.9) Gecko/20080705 Firefox/3.0 Kapiko/3.0', 'Mozilla/5.0 (X11; Linux i686; U;) Gecko/20070322 Firefox/59', 'Mozilla/5.0 (X11; U; Linux i686; en-US; rv:1.9.0.8) Gecko Fedora/1.9.0.8-1.fc10 Kazehakase/0.5.6', 'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/535.11 (KHTML, like Gecko) Chrome/17.0.963.56 Safari/535.11', 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_7_3) AppleWebKit/535.20 (KHTML, like Gecko) Chrome/19.0.1036.7 Safari/535.20', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/60.0.3112.113 Safari/537.36', 'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/60.0.3112.90 Safari/537.36', 'Mozilla/5.0 (Windows NT 5.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/60.0.3112.90 Safari/537.36', 'Mozilla/5.0 (Windows NT 6.2; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/60.0.3112.90 Safari/537.36', 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/44.0.2403.157 Safari/537.36', 'Mozilla/5.0 (Windows NT 6.3; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/60.0.3112.113 Safari/537.36', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/57.0.2987.133 Safari/537.36', 'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/57.0.2987.133 Safari/537.36', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/55.0.2883.87 Safari/537.36', 'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/55.0.2883.87 Safari/537.36', 'Mozilla/4.0 (compatible; MSIE 9.0; Windows NT 6.1)', 'Mozilla/5.0 (Windows NT 6.1; WOW64; Trident/7.0; rv:11.0) like Gecko', 'Mozilla/5.0 (compatible; MSIE 9.0; Windows NT 6.1; WOW64; Trident/5.0)', 'Mozilla/5.0 (Windows NT 6.1; Trident/7.0; rv:11.0) like Gecko', 'Mozilla/5.0 (Windows NT 6.2; WOW64; Trident/7.0; rv:11.0) like Gecko', 'Mozilla/5.0 (Windows NT 10.0; WOW64; Trident/7.0; rv:11.0) like Gecko', 'Mozilla/5.0 (compatible; MSIE 9.0; Windows NT 6.0; Trident/5.0)', 'Mozilla/5.0 (Windows NT 6.3; WOW64; Trident/7.0; rv:11.0) like Gecko', 'Mozilla/5.0 (compatible; MSIE 9.0; Windows NT 6.1; Trident/5.0)', 'Mozilla/5.0 (Windows NT 6.1; Win64; x64; Trident/7.0; rv:11.0) like Gecko', 'Mozilla/5.0 (compatible; MSIE 10.0; Windows NT 6.1; WOW64; Trident/6.0)']

def get_language(file_path):
    if False:
        for i in range(10):
            print('nop')
    'Returns the language a file is written in.'
    if file_path.endswith('.py'):
        return 'python3'
    elif file_path.endswith('.js'):
        return 'node'
    elif file_path.endswith('.go'):
        return 'go run'
    elif file_path.endswith('.rb'):
        return 'ruby'
    elif file_path.endswith('.java'):
        return 'javac'
    elif file_path.endswith('.class'):
        return 'java'
    else:
        return ''

def get_error_message(error, language):
    if False:
        print('Hello World!')
    'Filters the stack trace from stderr and returns only the error message.'
    if error == '':
        return None
    elif language == 'python3':
        if any((e in error for e in ['KeyboardInterrupt', 'SystemExit', 'GeneratorExit'])):
            return None
        else:
            return error.split('\n')[-2].strip()
    elif language == 'node':
        return error.split('\n')[4][1:]
    elif language == 'go run':
        return error.split('\n')[1].split(': ', 1)[1][1:]
    elif language == 'ruby':
        error_message = error.split('\n')[0]
        return error_message[error_message.rfind(': ') + 2:]
    elif language == 'javac':
        m = re.search('.*error:(.*)', error.split('\n')[0])
        return m.group(1) if m else None
    elif language == 'java':
        for line in error.split('\n'):
            m = re.search('.*(Exception|Error):(.*)', line)
            if m and m.group(2):
                return m.group(2)
            m = re.search('Exception in thread ".*" (.*)', line)
            if m and m.group(1):
                return m.group(1)
        return None

def read(pipe, funcs):
    if False:
        i = 10
        return i + 15
    'Reads and pushes piped output to a shared queue and appropriate lists.'
    for line in iter(pipe.readline, b''):
        for func in funcs:
            func(line.decode('utf-8'))
    pipe.close()

def write(get):
    if False:
        for i in range(10):
            print('nop')
    'Pulls output from shared queue and prints to terminal.'
    for line in iter(get, None):
        print(line)

def execute(command):
    if False:
        print('Hello World!')
    'Executes a given command and clones stdout/err to both variables and the\n    terminal (in real-time).'
    process = Popen(command, cwd=None, shell=False, close_fds=True, stdout=PIPE, stderr=PIPE, bufsize=1)
    (output, errors) = ([], [])
    pipe_queue = Queue()
    stdout_thread = Thread(target=read, args=(process.stdout, [pipe_queue.put, output.append]))
    stderr_thread = Thread(target=read, args=(process.stderr, [pipe_queue.put, errors.append]))
    writer_thread = Thread(target=write, args=(pipe_queue.get,))
    for thread in (stdout_thread, stderr_thread, writer_thread):
        thread.daemon = True
        thread.start()
    process.wait()
    for thread in (stdout_thread, stderr_thread):
        thread.join()
    pipe_queue.put(None)
    output = ' '.join(output)
    errors = ' '.join(errors)
    if 'java' != command[0] and (not os.path.isfile(command[1])):
        return (None, None)
    else:
        return (output, errors)

def stylize_code(soup):
    if False:
        print('Hello World!')
    'Identifies and stylizes code in a question or answer.'
    stylized_text = []
    code_blocks = [block.get_text() for block in soup.find_all('code')]
    blockquotes = [block.get_text() for block in soup.find_all('blockquote')]
    newline = False
    for child in soup.recursiveChildGenerator():
        name = getattr(child, 'name', None)
        if name is None:
            if child in code_blocks:
                if newline:
                    stylized_text.append(('code', u'\n%s' % str(child)))
                    newline = False
                else:
                    stylized_text.append(('code', u'%s' % str(child)))
            else:
                newline = child.endswith('\n')
                stylized_text.append(u'%s' % str(child))
    if type(stylized_text[-2]) == tuple:
        if stylized_text[-2][1].endswith('\n'):
            stylized_text[-2] = ('code', stylized_text[-2][1][:-1])
    return urwid.Text(stylized_text)

def get_search_results(soup):
    if False:
        while True:
            i = 10
    'Returns a list of dictionaries containing each search result.'
    search_results = []
    for result in soup.find_all('div', class_='question-summary search-result'):
        title_container = result.find_all('div', class_='result-link')[0].find_all('a')[0]
        if result.find_all('div', class_='status answered') != []:
            answer_count = int(result.find_all('div', class_='status answered')[0].find_all('strong')[0].text)
        elif result.find_all('div', class_='status answered-accepted') != []:
            answer_count = int(result.find_all('div', class_='status answered-accepted')[0].find_all('strong')[0].text)
        else:
            answer_count = 0
        search_results.append({'Title': title_container['title'], 'Answers': answer_count, 'URL': SO_URL + title_container['href']})
    return search_results

def souper(url):
    if False:
        while True:
            i = 10
    'Turns a given URL into a BeautifulSoup object.'
    try:
        html = requests.get(url, headers={'User-Agent': random.choice(USER_AGENTS)})
    except requests.exceptions.RequestException:
        sys.stdout.write('\n%s%s%s' % (RED, 'Rebound was unable to fetch Stack Overflow results. Please check that you are connected to the internet.\n', END))
        sys.exit(1)
    if re.search('\\.com/nocaptcha', html.url):
        return None
    else:
        return BeautifulSoup(html.text, 'html.parser')

def search_stackoverflow(query):
    if False:
        while True:
            i = 10
    'Wrapper function for get_search_results.'
    soup = souper(SO_URL + '/search?pagesize=50&q=%s' % query.replace(' ', '+'))
    if soup == None:
        return (None, True)
    else:
        return (get_search_results(soup), False)

def get_question_and_answers(url):
    if False:
        i = 10
        return i + 15
    'Returns details about a given question and list of its answers.'
    soup = souper(url)
    if soup == None:
        return ('Sorry, Stack Overflow blocked our request. Try again in a couple seconds.', '', '', '')
    else:
        question_title = soup.find_all('a', class_='question-hyperlink')[0].get_text()
        question_stats = soup.find('div', attrs={'itemprop': 'upvoteCount'}).get_text()
        question_stats += ' Votes | Asked ' + soup.find('time', attrs={'itemprop': 'dateCreated'}).get_text()
        question_desc = stylize_code(soup.find_all('div', class_='s-prose js-post-body')[0])
        answers = [stylize_code(answer) for answer in soup.find_all('div', class_='s-prose js-post-body')][1:]
        if len(answers) == 0:
            answers.append(urwid.Text(('no answers', u'\nNo answers for this question.')))
        return (question_title, question_desc, question_stats, answers)

class Scrollable(urwid.WidgetDecoration):

    def sizing(self):
        if False:
            print('Hello World!')
        return frozenset([BOX])

    def selectable(self):
        if False:
            while True:
                i = 10
        return True

    def __init__(self, widget):
        if False:
            for i in range(10):
                print('nop')
        'Box widget (wrapper) that makes a fixed or flow widget vertically scrollable.'
        self._trim_top = 0
        self._scroll_action = None
        self._forward_keypress = None
        self._old_cursor_coords = None
        self._rows_max_cached = 0
        self._rows_max_displayable = 0
        self.__super.__init__(widget)

    def render(self, size, focus=False):
        if False:
            while True:
                i = 10
        (maxcol, maxrow) = size
        ow = self._original_widget
        ow_size = self._get_original_widget_size(size)
        canv = urwid.CompositeCanvas(ow.render(ow_size, focus))
        (canv_cols, canv_rows) = (canv.cols(), canv.rows())
        if canv_cols <= maxcol:
            pad_width = maxcol - canv_cols
            if pad_width > 0:
                canv.pad_trim_left_right(0, pad_width)
        if canv_rows <= maxrow:
            fill_height = maxrow - canv_rows
            if fill_height > 0:
                canv.pad_trim_top_bottom(0, fill_height)
        self._rows_max_displayable = maxrow
        if canv_cols <= maxcol and canv_rows <= maxrow:
            return canv
        self._adjust_trim_top(canv, size)
        trim_top = self._trim_top
        trim_end = canv_rows - maxrow - trim_top
        trim_right = canv_cols - maxcol
        if trim_top > 0:
            canv.trim(trim_top)
        if trim_end > 0:
            canv.trim_end(trim_end)
        if trim_right > 0:
            canv.pad_trim_left_right(0, -trim_right)
        if canv.cursor is not None:
            (curscol, cursrow) = canv.cursor
            if cursrow >= maxrow or cursrow < 0:
                canv.cursor = None
        self._forward_keypress = bool(canv.cursor)
        return canv

    def keypress(self, size, key):
        if False:
            for i in range(10):
                print('nop')
        if self._forward_keypress:
            ow = self._original_widget
            ow_size = self._get_original_widget_size(size)
            if hasattr(ow, 'get_cursor_coords'):
                self._old_cursor_coords = ow.get_cursor_coords(ow_size)
            key = ow.keypress(ow_size, key)
            if key is None:
                return None
        command_map = self._command_map
        if command_map[key] == urwid.CURSOR_UP:
            self._scroll_action = SCROLL_LINE_UP
        elif command_map[key] == urwid.CURSOR_DOWN:
            self._scroll_action = SCROLL_LINE_DOWN
        elif command_map[key] == urwid.CURSOR_PAGE_UP:
            self._scroll_action = SCROLL_PAGE_UP
        elif command_map[key] == urwid.CURSOR_PAGE_DOWN:
            self._scroll_action = SCROLL_PAGE_DOWN
        elif command_map[key] == urwid.CURSOR_MAX_LEFT:
            self._scroll_action = SCROLL_TO_TOP
        elif command_map[key] == urwid.CURSOR_MAX_RIGHT:
            self._scroll_action = SCROLL_TO_END
        else:
            return key
        self._invalidate()

    def mouse_event(self, size, event, button, col, row, focus):
        if False:
            return 10
        ow = self._original_widget
        if hasattr(ow, 'mouse_event'):
            ow_size = self._get_original_widget_size(size)
            row += self._trim_top
            return ow.mouse_event(ow_size, event, button, col, row, focus)
        else:
            return False

    def _adjust_trim_top(self, canv, size):
        if False:
            return 10
        'Adjust self._trim_top according to self._scroll_action'
        action = self._scroll_action
        self._scroll_action = None
        (maxcol, maxrow) = size
        trim_top = self._trim_top
        canv_rows = canv.rows()
        if trim_top < 0:
            trim_top = canv_rows - maxrow + trim_top + 1
        if canv_rows <= maxrow:
            self._trim_top = 0
            return

        def ensure_bounds(new_trim_top):
            if False:
                print('Hello World!')
            return max(0, min(canv_rows - maxrow, new_trim_top))
        if action == SCROLL_LINE_UP:
            self._trim_top = ensure_bounds(trim_top - 1)
        elif action == SCROLL_LINE_DOWN:
            self._trim_top = ensure_bounds(trim_top + 1)
        elif action == SCROLL_PAGE_UP:
            self._trim_top = ensure_bounds(trim_top - maxrow + 1)
        elif action == SCROLL_PAGE_DOWN:
            self._trim_top = ensure_bounds(trim_top + maxrow - 1)
        elif action == SCROLL_TO_TOP:
            self._trim_top = 0
        elif action == SCROLL_TO_END:
            self._trim_top = canv_rows - maxrow
        else:
            self._trim_top = ensure_bounds(trim_top)
        if self._old_cursor_coords is not None and self._old_cursor_coords != canv.cursor:
            self._old_cursor_coords = None
            (curscol, cursrow) = canv.cursor
            if cursrow < self._trim_top:
                self._trim_top = cursrow
            elif cursrow >= self._trim_top + maxrow:
                self._trim_top = max(0, cursrow - maxrow + 1)

    def _get_original_widget_size(self, size):
        if False:
            return 10
        ow = self._original_widget
        sizing = ow.sizing()
        if FIXED in sizing:
            return ()
        elif FLOW in sizing:
            return (size[0],)

    def get_scrollpos(self, size=None, focus=False):
        if False:
            for i in range(10):
                print('nop')
        return self._trim_top

    def set_scrollpos(self, position):
        if False:
            return 10
        self._trim_top = int(position)
        self._invalidate()

    def rows_max(self, size=None, focus=False):
        if False:
            i = 10
            return i + 15
        if size is not None:
            ow = self._original_widget
            ow_size = self._get_original_widget_size(size)
            sizing = ow.sizing()
            if FIXED in sizing:
                self._rows_max_cached = ow.pack(ow_size, focus)[1]
            elif FLOW in sizing:
                self._rows_max_cached = ow.rows(ow_size, focus)
            else:
                raise RuntimeError('Not a flow/box widget: %r' % self._original_widget)
        return self._rows_max_cached

    @property
    def scroll_ratio(self):
        if False:
            i = 10
            return i + 15
        return self._rows_max_cached / self._rows_max_displayable

class ScrollBar(urwid.WidgetDecoration):

    def sizing(self):
        if False:
            return 10
        return frozenset((BOX,))

    def selectable(self):
        if False:
            while True:
                i = 10
        return True

    def __init__(self, widget, thumb_char=u'█', trough_char=' ', side=SCROLLBAR_RIGHT, width=1):
        if False:
            print('Hello World!')
        'Box widget that adds a scrollbar to `widget`.'
        self.__super.__init__(widget)
        self._thumb_char = thumb_char
        self._trough_char = trough_char
        self.scrollbar_side = side
        self.scrollbar_width = max(1, width)
        self._original_widget_size = (0, 0)
        self._dragging = False

    def render(self, size, focus=False):
        if False:
            return 10
        (maxcol, maxrow) = size
        ow = self._original_widget
        ow_base = self.scrolling_base_widget
        ow_rows_max = ow_base.rows_max(size, focus)
        if ow_rows_max <= maxrow:
            self._original_widget_size = size
            return ow.render(size, focus)
        sb_width = self._scrollbar_width
        self._original_widget_size = ow_size = (maxcol - sb_width, maxrow)
        ow_canv = ow.render(ow_size, focus)
        pos = ow_base.get_scrollpos(ow_size, focus)
        posmax = ow_rows_max - maxrow
        thumb_weight = min(1, maxrow / max(1, ow_rows_max))
        thumb_height = max(1, round(thumb_weight * maxrow))
        top_weight = float(pos) / max(1, posmax)
        top_height = int((maxrow - thumb_height) * top_weight)
        if top_height == 0 and top_weight > 0:
            top_height = 1
        bottom_height = maxrow - thumb_height - top_height
        assert thumb_height + top_height + bottom_height == maxrow
        top = urwid.SolidCanvas(self._trough_char, sb_width, top_height)
        thumb = urwid.SolidCanvas(self._thumb_char, sb_width, thumb_height)
        bottom = urwid.SolidCanvas(self._trough_char, sb_width, bottom_height)
        sb_canv = urwid.CanvasCombine([(top, None, False), (thumb, None, False), (bottom, None, False)])
        combinelist = [(ow_canv, None, True, ow_size[0]), (sb_canv, None, False, sb_width)]
        if self._scrollbar_side != SCROLLBAR_LEFT:
            return urwid.CanvasJoin(combinelist)
        else:
            return urwid.CanvasJoin(reversed(combinelist))

    @property
    def scrollbar_width(self):
        if False:
            print('Hello World!')
        return max(1, self._scrollbar_width)

    @scrollbar_width.setter
    def scrollbar_width(self, width):
        if False:
            i = 10
            return i + 15
        self._scrollbar_width = max(1, int(width))
        self._invalidate()

    @property
    def scrollbar_side(self):
        if False:
            while True:
                i = 10
        return self._scrollbar_side

    @scrollbar_side.setter
    def scrollbar_side(self, side):
        if False:
            print('Hello World!')
        if side not in (SCROLLBAR_LEFT, SCROLLBAR_RIGHT):
            raise ValueError("scrollbar_side must be 'left' or 'right', not %r" % side)
        self._scrollbar_side = side
        self._invalidate()

    @property
    def scrolling_base_widget(self):
        if False:
            return 10
        'Nearest `base_widget` that is compatible with the scrolling API.'

        def orig_iter(w):
            if False:
                return 10
            while hasattr(w, 'original_widget'):
                w = w.original_widget
                yield w
            yield w

        def is_scrolling_widget(w):
            if False:
                print('Hello World!')
            return hasattr(w, 'get_scrollpos') and hasattr(w, 'rows_max')
        for w in orig_iter(self):
            if is_scrolling_widget(w):
                return w

    @property
    def scrollbar_column(self):
        if False:
            print('Hello World!')
        if self.scrollbar_side == SCROLLBAR_LEFT:
            return 0
        if self.scrollbar_side == SCROLLBAR_RIGHT:
            return self._original_widget_size[0]

    def keypress(self, size, key):
        if False:
            i = 10
            return i + 15
        return self._original_widget.keypress(self._original_widget_size, key)

    def mouse_event(self, size, event, button, col, row, focus):
        if False:
            print('Hello World!')
        ow = self._original_widget
        ow_size = self._original_widget_size
        handled = False
        if hasattr(ow, 'mouse_event'):
            handled = ow.mouse_event(ow_size, event, button, col, row, focus)
        if not handled and hasattr(ow, 'set_scrollpos'):
            if button == 4:
                pos = ow.get_scrollpos(ow_size)
                if pos > 0:
                    ow.set_scrollpos(pos - 1)
                    return True
            elif button == 5:
                pos = ow.get_scrollpos(ow_size)
                ow.set_scrollpos(pos + 1)
                return True
            elif col == self.scrollbar_column:
                ow.set_scrollpos(int(row * ow.scroll_ratio))
                if event == 'mouse press':
                    self._dragging = True
                elif event == 'mouse release':
                    self._dragging = False
            elif self._dragging:
                ow.set_scrollpos(int(row * ow.scroll_ratio))
                if event == 'mouse release':
                    self._dragging = False
        return False

class SelectableText(urwid.Text):

    def selectable(self):
        if False:
            i = 10
            return i + 15
        return True

    def keypress(self, size, key):
        if False:
            while True:
                i = 10
        return key

def interleave(a, b):
    if False:
        for i in range(10):
            print('nop')
    result = []
    while a and b:
        result.append(a.pop(0))
        result.append(b.pop(0))
    result.extend(a)
    result.extend(b)
    return result

class App(object):

    def __init__(self, search_results):
        if False:
            return 10
        (self.search_results, self.viewing_answers) = (search_results, False)
        self.palette = [('title', 'light cyan,bold', 'default', 'standout'), ('stats', 'light green', 'default', 'standout'), ('menu', 'black', 'light cyan', 'standout'), ('reveal focus', 'black', 'light cyan', 'standout'), ('reveal viewed focus', 'yellow, bold', 'light cyan', 'standout'), ('no answers', 'light red', 'default', 'standout'), ('code', 'brown', 'default', 'standout'), ('viewed', 'yellow', 'default', 'standout')]
        self.menu = urwid.Text([u'\n', ('menu', u' ENTER '), ('light gray', u' View answers '), ('menu', u' B '), ('light gray', u' Open browser '), ('menu', u' Q '), ('light gray', u' Quit')])
        results = list(map(lambda result: urwid.AttrMap(SelectableText(self._stylize_title(result)), None, 'reveal focus'), self.search_results))
        self.content = urwid.SimpleListWalker(results)
        self.content_container = urwid.ListBox(self.content)
        layout = urwid.Frame(body=self.content_container, footer=self.menu)
        self.main_loop = urwid.MainLoop(layout, self.palette, unhandled_input=self._handle_input)
        self.original_widget = self.main_loop.widget
        self.main_loop.run()

    def _handle_input(self, input):
        if False:
            print('Hello World!')
        if input == 'enter' or (input[0] == 'meta mouse press' and input[1] == 1):
            url = self._get_selected_link()
            if url != None:
                self.viewing_answers = True
                (question_title, question_desc, question_stats, answers) = get_question_and_answers(url)
                pile = urwid.Pile(self._stylize_question(question_title, question_desc, question_stats) + [urwid.Divider('*')] + interleave(answers, [urwid.Divider('-')] * (len(answers) - 1)))
                padding = ScrollBar(Scrollable(urwid.Padding(pile, left=2, right=2)))
                linebox = urwid.LineBox(padding)
                menu = urwid.Text([u'\n', ('menu', u' ESC '), ('light gray', u' Go back '), ('menu', u' B '), ('light gray', u' Open browser '), ('menu', u' Q '), ('light gray', u' Quit')])
                (_, idx) = self.content_container.get_focus()
                txt = self.content[idx].original_widget.text
                self.content[idx] = urwid.AttrMap(SelectableText(txt), 'viewed', 'reveal viewed focus')
                self.main_loop.widget = urwid.Frame(body=urwid.Overlay(linebox, self.content_container, 'center', ('relative', 60), 'middle', 23), footer=menu)
        elif input in ('b', 'B') or (input[0] == 'ctrl mouse press' and input[1] == 1):
            url = self._get_selected_link()
            if url != None:
                webbrowser.open(url)
        elif input == 'esc':
            if self.viewing_answers:
                self.main_loop.widget = self.original_widget
                self.viewing_answers = False
            else:
                raise urwid.ExitMainLoop()
        elif input in ('q', 'Q'):
            raise urwid.ExitMainLoop()

    def _get_selected_link(self):
        if False:
            print('Hello World!')
        (focus_widget, idx) = self.content_container.get_focus()
        title = focus_widget.base_widget.text
        for result in self.search_results:
            if title == self._stylize_title(result):
                return result['URL']

    def _stylize_title(self, search_result):
        if False:
            for i in range(10):
                print('nop')
        if search_result['Answers'] == 1:
            return '%s (1 Answer)' % search_result['Title']
        else:
            return '%s (%s Answers)' % (search_result['Title'], search_result['Answers'])

    def _stylize_question(self, title, desc, stats):
        if False:
            while True:
                i = 10
        new_title = urwid.Text(('title', u'%s' % title))
        new_stats = urwid.Text(('stats', u'%s\n' % stats))
        return [new_title, desc, new_stats]

def confirm(question):
    if False:
        for i in range(10):
            print('nop')
    'Prompts a given question and handles user input.'
    valid = {'yes': True, 'y': True, 'ye': True, 'no': False, 'n': False, '': True}
    prompt = ' [Y/n] '
    while True:
        print(BOLD + CYAN + question + prompt + END)
        choice = input().lower()
        if choice in valid:
            return valid[choice]
        print("Please respond with 'yes' or 'no' (or 'y' or 'n').\n")

def print_help():
    if False:
        for i in range(10):
            print('nop')
    'Prints usage instructions.'
    print('%sRebound, V1.1.9a1 - Made by @shobrook%s\n' % (BOLD, END))
    print('Command-line tool that automatically searches Stack Overflow and displays results in your terminal when you get a compiler error.')
    print('\n\n%sUsage:%s $ rebound %s[file_name]%s\n' % (UNDERLINE, END, YELLOW, END))
    print('\n$ python3 %stest.py%s   =>   $ rebound %stest.py%s' % (YELLOW, END, YELLOW, END))
    print('\n$ node %stest.js%s     =>   $ rebound %stest.js%s\n' % (YELLOW, END, YELLOW, END))
    print('\nIf you just want to query Stack Overflow, use the -q parameter: $ rebound -q %sWhat is an array comprehension?%s\n\n' % (YELLOW, END))

def main():
    if False:
        while True:
            i = 10
    if len(sys.argv) == 1 or sys.argv[1].lower() == '-h' or sys.argv[1].lower() == '--help':
        print_help()
    elif sys.argv[1].lower() == '-q' or sys.argv[1].lower() == '--query':
        query = ' '.join(sys.argv[2:])
        (search_results, captcha) = search_stackoverflow(query)
        if search_results != []:
            if captcha:
                print('\n%s%s%s' % (RED, 'Sorry, Stack Overflow blocked our request. Try again in a minute.\n', END))
                return
            else:
                App(search_results)
        else:
            print('\n%s%s%s' % (RED, 'No Stack Overflow results found.\n', END))
    else:
        language = get_language(sys.argv[1].lower())
        if language == '':
            print('\n%s%s%s' % (RED, "Sorry, Rebound doesn't support this file type.\n", END))
            return
        file_path = sys.argv[1:]
        if language == 'java':
            file_path = [f.replace('.class', '') for f in file_path]
        (output, error) = execute([language] + file_path)
        if (output, error) == (None, None):
            return
        error_msg = get_error_message(error, language)
        if error_msg != None:
            language = 'java' if language == 'javac' else language
            query = '%s %s' % (language, error_msg)
            (search_results, captcha) = search_stackoverflow(query)
            if search_results != []:
                if captcha:
                    print('\n%s%s%s' % (RED, 'Sorry, Stack Overflow blocked our request. Try again in a minute.\n', END))
                    return
                elif confirm('\nDisplay Stack Overflow results?'):
                    App(search_results)
            else:
                print('\n%s%s%s' % (RED, 'No Stack Overflow results found.\n', END))
        else:
            print('\n%s%s%s' % (CYAN, 'No error detected :)\n', END))
    return