"""Curses browser interface class ."""
import math
import curses
from glances.outputs.glances_curses import _GlancesCurses
from glances.logger import logger
from glances.timer import Timer

class GlancesCursesBrowser(_GlancesCurses):
    """Class for the Glances curse client browser."""

    def __init__(self, args=None):
        if False:
            print('Hello World!')
        'Init the father class.'
        super(GlancesCursesBrowser, self).__init__(args=args)
        _colors_list = {'UNKNOWN': self.no_color, 'SNMP': self.default_color2, 'ONLINE': self.default_color2, 'OFFLINE': self.ifCRITICAL_color2, 'PROTECTED': self.ifWARNING_color2}
        self.colors_list.update(_colors_list)
        self.first_scan = True
        self.__refresh_time = args.time
        self.cursor_position = 0
        self._active_server = None
        self._current_page = 0
        self._page_max = 0
        self._page_max_lines = 0
        self.is_end = False
        self._revesed_sorting = False
        self._stats_list = None

    @property
    def active_server(self):
        if False:
            for i in range(10):
                print('nop')
        "Return the active server or None if it's the browser list."
        return self._active_server

    @active_server.setter
    def active_server(self, index):
        if False:
            for i in range(10):
                print('nop')
        'Set the active server or None if no server selected.'
        self._active_server = index

    @property
    def cursor(self):
        if False:
            while True:
                i = 10
        'Get the cursor position.'
        return self.cursor_position

    @cursor.setter
    def cursor(self, position):
        if False:
            return 10
        'Set the cursor position.'
        self.cursor_position = position

    def get_pagelines(self, stats):
        if False:
            i = 10
            return i + 15
        if self._current_page == self._page_max - 1:
            page_lines = len(stats) % self._page_max_lines
        else:
            page_lines = self._page_max_lines
        return page_lines

    def _get_status_count(self, stats):
        if False:
            return 10
        counts = {}
        for item in stats:
            color = item['status']
            counts[color] = counts.get(color, 0) + 1
        result = ''
        for key in counts.keys():
            result += key + ': ' + str(counts[key]) + ' '
        return result

    def _get_stats(self, stats):
        if False:
            while True:
                i = 10
        stats_list = None
        if self._stats_list is not None:
            stats_list = self._stats_list
            stats_list.sort(reverse=self._revesed_sorting, key=lambda x: {'UNKNOWN': 0, 'OFFLINE': 1, 'PROTECTED': 2, 'SNMP': 3, 'ONLINE': 4}.get(x['status'], 99))
        else:
            stats_list = stats
        return stats_list

    def cursor_up(self, stats):
        if False:
            print('Hello World!')
        'Set the cursor to position N-1 in the list.'
        if 0 <= self.cursor_position - 1:
            self.cursor_position -= 1
        elif self._current_page - 1 < 0:
            self._current_page = self._page_max - 1
            self.cursor_position = (len(stats) - 1) % self._page_max_lines
        else:
            self._current_page -= 1
            self.cursor_position = self._page_max_lines - 1

    def cursor_down(self, stats):
        if False:
            while True:
                i = 10
        'Set the cursor to position N-1 in the list.'
        if self.cursor_position + 1 < self.get_pagelines(stats):
            self.cursor_position += 1
        else:
            if self._current_page + 1 < self._page_max:
                self._current_page += 1
            else:
                self._current_page = 0
            self.cursor_position = 0

    def cursor_pageup(self, stats):
        if False:
            i = 10
            return i + 15
        'Set prev page.'
        if self._current_page - 1 < 0:
            self._current_page = self._page_max - 1
        else:
            self._current_page -= 1
        self.cursor_position = 0

    def cursor_pagedown(self, stats):
        if False:
            i = 10
            return i + 15
        'Set next page.'
        if self._current_page + 1 < self._page_max:
            self._current_page += 1
        else:
            self._current_page = 0
        self.cursor_position = 0

    def __catch_key(self, stats):
        if False:
            while True:
                i = 10
        self.pressedkey = self.get_key(self.term_window)
        refresh = False
        if self.pressedkey != -1:
            logger.debug('Key pressed. Code=%s' % self.pressedkey)
        if self.pressedkey == ord('\x1b') or self.pressedkey == ord('q'):
            self.end()
            logger.info('Stop Glances client browser')
            self.is_end = True
        elif self.pressedkey == 10:
            self.active_server = self._current_page * self._page_max_lines + self.cursor_position
            logger.debug('Server {}/{} selected'.format(self.active_server, len(stats)))
        elif self.pressedkey == curses.KEY_UP or self.pressedkey == 65:
            self.cursor_up(stats)
            logger.debug('Server {}/{} selected'.format(self.cursor + 1, len(stats)))
        elif self.pressedkey == curses.KEY_DOWN or self.pressedkey == 66:
            self.cursor_down(stats)
            logger.debug('Server {}/{} selected'.format(self.cursor + 1, len(stats)))
        elif self.pressedkey == curses.KEY_PPAGE:
            self.cursor_pageup(stats)
            logger.debug('PageUP: Server ({}/{}) pages.'.format(self._current_page + 1, self._page_max))
        elif self.pressedkey == curses.KEY_NPAGE:
            self.cursor_pagedown(stats)
            logger.debug('PageDown: Server {}/{} pages'.format(self._current_page + 1, self._page_max))
        elif self.pressedkey == ord('1'):
            self._stats_list = None
            refresh = True
        elif self.pressedkey == ord('2'):
            self._revesed_sorting = False
            self._stats_list = stats.copy()
            refresh = True
        elif self.pressedkey == ord('3'):
            self._revesed_sorting = True
            self._stats_list = stats.copy()
            refresh = True
        if refresh:
            self._current_page = 0
            self.cursor_position = 0
            self.flush(stats)
        return self.pressedkey

    def update(self, stats, duration=3, cs_status=None, return_to_browser=False):
        if False:
            i = 10
            return i + 15
        "Update the servers' list screen.\n\n        Wait for __refresh_time sec / catch key every 100 ms.\n\n        :param stats: Dict of dict with servers stats\n        :param cs_status:\n        :param duration:\n        :param return_to_browser:\n        "
        logger.debug('Servers list: {}'.format(stats))
        self.flush(stats)
        exitkey = False
        countdown = Timer(self.__refresh_time)
        while not countdown.finished() and (not exitkey):
            pressedkey = self.__catch_key(stats)
            exitkey = pressedkey == ord('\x1b') or pressedkey == ord('q') or pressedkey == 10
            if not exitkey and pressedkey > -1:
                self.flush(stats)
            self.wait()
        return self.active_server

    def flush(self, stats):
        if False:
            for i in range(10):
                print('nop')
        "Update the servers' list screen.\n\n        :param stats: List of dict with servers stats\n        "
        self.erase()
        self.display(stats)

    def display(self, stats, cs_status=None):
        if False:
            return 10
        'Display the servers list.\n\n        :return: True if the stats have been displayed else False (no server available)\n        '
        self.init_line_column()
        screen_x = self.screen.getmaxyx()[1]
        screen_y = self.screen.getmaxyx()[0]
        stats_max = screen_y - 3
        stats_len = len(stats)
        self._page_max_lines = stats_max
        self._page_max = int(math.ceil(stats_len / stats_max))
        x = 0
        y = 0
        if stats_len == 0:
            if self.first_scan and (not self.args.disable_autodiscover):
                msg = 'Glances is scanning your network. Please wait...'
                self.first_scan = False
            else:
                msg = 'No Glances server available'
        elif len(stats) == 1:
            msg = 'One Glances server available'
        else:
            msg = '{} Glances servers available'.format(stats_len)
        if self.args.disable_autodiscover:
            msg += ' (auto discover is disabled)'
        if screen_y > 1:
            self.term_window.addnstr(y, x, msg, screen_x - x, self.colors_list['TITLE'])
            msg = '{}'.format(self._get_status_count(stats))
            self.term_window.addnstr(y + 1, x, msg, screen_x - x)
        if stats_len > stats_max and screen_y > 2:
            msg = '{} servers displayed.({}/{}) {}'.format(self.get_pagelines(stats), self._current_page + 1, self._page_max, self._get_status_count(stats))
            self.term_window.addnstr(y + 1, x, msg, screen_x - x)
        if stats_len == 0:
            return False
        column_def = [['name', 'Name', 16], ['alias', None, None], ['load_min5', 'LOAD', 6], ['cpu_percent', 'CPU%', 5], ['mem_percent', 'MEM%', 5], ['status', 'STATUS', 9], ['ip', 'IP', 15], ['hr_name', 'OS', 16]]
        y = 2
        xc = x + 2
        for (cpt, c) in enumerate(column_def):
            if xc < screen_x and y < screen_y and (c[1] is not None):
                self.term_window.addnstr(y, xc, c[1], screen_x - x, self.colors_list['BOLD'])
                xc += c[2] + self.space_between_column
        y += 1
        if self.cursor > len(stats) - 1:
            self.cursor = len(stats) - 1
        stats_list = self._get_stats(stats)
        start_line = self._page_max_lines * self._current_page
        end_line = start_line + self.get_pagelines(stats_list)
        current_page = stats_list[start_line:end_line]
        line = 0
        for v in current_page:
            if line >= stats_max:
                continue
            server_stat = {}
            for c in column_def:
                try:
                    server_stat[c[0]] = v[c[0]]
                except KeyError as e:
                    logger.debug('Cannot grab stats {} from server (KeyError: {})'.format(c[0], e))
                    server_stat[c[0]] = '?'
                try:
                    if c[0] == 'alias' and v[c[0]] is not None:
                        server_stat['name'] = v[c[0]]
                except KeyError:
                    pass
            cpt = 0
            xc = x
            if line == self.cursor:
                self.term_window.addnstr(y, xc, '>', screen_x - xc, self.colors_list['BOLD'])
            xc += 2
            for c in column_def:
                if xc < screen_x and y < screen_y and (c[1] is not None):
                    self.term_window.addnstr(y, xc, format(server_stat[c[0]]), c[2], self.colors_list[v['status']])
                    xc += c[2] + self.space_between_column
                cpt += 1
            y += 1
            line += 1
        return True