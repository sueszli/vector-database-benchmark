"""Curses interface class."""
from __future__ import unicode_literals
import sys
from glances.globals import MACOS, WINDOWS, nativestr, u, itervalues, enable, disable
from glances.logger import logger
from glances.events import glances_events
from glances.processes import glances_processes, sort_processes_key_list
from glances.outputs.glances_unicode import unicode_message
from glances.timer import Timer
try:
    import curses
    import curses.panel
    from curses.textpad import Textbox
except ImportError:
    logger.critical('Curses module not found. Glances cannot start in standalone mode.')
    if WINDOWS:
        logger.critical('For Windows you can try installing windows-curses with pip install.')
    sys.exit(1)

class _GlancesCurses(object):
    """This class manages the curses display (and key pressed).

    Note: It is a private class, use GlancesCursesClient or GlancesCursesBrowser.
    """
    _hotkeys = {'0': {'switch': 'disable_irix'}, '1': {'switch': 'percpu'}, '2': {'switch': 'disable_left_sidebar'}, '3': {'switch': 'disable_quicklook'}, '6': {'switch': 'meangpu'}, '9': {'switch': 'theme_white'}, '/': {'switch': 'process_short_name'}, 'a': {'sort_key': 'auto'}, 'A': {'switch': 'disable_amps'}, 'b': {'switch': 'byte'}, 'B': {'switch': 'diskio_iops'}, 'c': {'sort_key': 'cpu_percent'}, 'C': {'switch': 'disable_cloud'}, 'd': {'switch': 'disable_diskio'}, 'D': {'switch': 'disable_containers'}, 'F': {'switch': 'fs_free_space'}, 'g': {'switch': 'generate_graph'}, 'G': {'switch': 'disable_gpu'}, 'h': {'switch': 'help_tag'}, 'i': {'sort_key': 'io_counters'}, 'I': {'switch': 'disable_ip'}, 'j': {'switch': 'programs'}, 'K': {'switch': 'disable_connections'}, 'l': {'switch': 'disable_alert'}, 'm': {'sort_key': 'memory_percent'}, 'M': {'switch': 'reset_minmax_tag'}, 'n': {'switch': 'disable_network'}, 'N': {'switch': 'disable_now'}, 'p': {'sort_key': 'name'}, 'P': {'switch': 'disable_ports'}, 'Q': {'switch': 'enable_irq'}, 'r': {'switch': 'disable_smart'}, 'R': {'switch': 'disable_raid'}, 's': {'switch': 'disable_sensors'}, 'S': {'switch': 'sparkline'}, 't': {'sort_key': 'cpu_times'}, 'T': {'switch': 'network_sum'}, 'u': {'sort_key': 'username'}, 'U': {'switch': 'network_cumul'}, 'W': {'switch': 'disable_wifi'}}
    _sort_loop = sort_processes_key_list
    _top = ['quicklook', 'cpu', 'percpu', 'gpu', 'mem', 'memswap', 'load']
    _quicklook_max_width = 68
    _left_sidebar = ['network', 'wifi', 'connections', 'ports', 'diskio', 'fs', 'irq', 'folders', 'raid', 'smart', 'sensors', 'now']
    _left_sidebar_min_width = 23
    _left_sidebar_max_width = 34
    _right_sidebar = ['containers', 'processcount', 'amps', 'processlist', 'alert']

    def __init__(self, config=None, args=None):
        if False:
            for i in range(10):
                print('nop')
        self.config = config
        self.args = args
        self.term_w = 80
        self.term_h = 24
        self.space_between_column = 3
        self.space_between_line = 2
        try:
            self.screen = curses.initscr()
            if not self.screen:
                logger.critical('Cannot init the curses library.\n')
                sys.exit(1)
            else:
                logger.debug('Curses library initialized with term: {}'.format(curses.longname()))
        except Exception as e:
            if args.export:
                logger.info('Cannot init the curses library, quiet mode on and export.')
                args.quiet = True
                return
            else:
                logger.critical('Cannot init the curses library ({})'.format(e))
                sys.exit(1)
        self.theme = {'name': 'black'}
        self.load_config(config)
        self._init_cursor()
        self._init_colors()
        self.term_window = self.screen.subwin(0, 0)
        self.edit_filter = False
        self.increase_nice_process = False
        self.decrease_nice_process = False
        self.kill_process = False
        self.args.reset_minmax_tag = False
        self.args.cursor_position = 0
        self.term_window.keypad(1)
        self.term_window.nodelay(1)
        self.pressedkey = -1
        self._init_history()

    def load_config(self, config):
        if False:
            return 10
        'Load the outputs section of the configuration file.'
        if config is not None and config.has_section('outputs'):
            logger.debug('Read the outputs section in the configuration file')
            self.theme['name'] = config.get_value('outputs', 'curse_theme', default='black')
            logger.debug('Theme for the curse interface: {}'.format(self.theme['name']))

    def is_theme(self, name):
        if False:
            return 10
        'Return True if the theme *name* should be used.'
        return getattr(self.args, 'theme_' + name) or self.theme['name'] == name

    def _init_history(self):
        if False:
            i = 10
            return i + 15
        'Init the history option.'
        self.reset_history_tag = False

    def _init_cursor(self):
        if False:
            i = 10
            return i + 15
        'Init cursors.'
        if hasattr(curses, 'noecho'):
            curses.noecho()
        if hasattr(curses, 'cbreak'):
            curses.cbreak()
        self.set_cursor(0)

    def _init_colors(self):
        if False:
            return 10
        'Init the Curses color layout.'
        try:
            if hasattr(curses, 'start_color'):
                curses.start_color()
                logger.debug('Curses interface compatible with {} colors'.format(curses.COLORS))
            if hasattr(curses, 'use_default_colors'):
                curses.use_default_colors()
        except Exception as e:
            logger.warning('Error initializing terminal color ({})'.format(e))
        if self.args.disable_bold:
            A_BOLD = 0
            self.args.disable_bg = True
        else:
            A_BOLD = curses.A_BOLD
        self.title_color = A_BOLD
        self.title_underline_color = A_BOLD | curses.A_UNDERLINE
        self.help_color = A_BOLD
        if curses.has_colors():
            if self.is_theme('white'):
                curses.init_pair(1, curses.COLOR_BLACK, -1)
            else:
                curses.init_pair(1, curses.COLOR_WHITE, -1)
            if self.args.disable_bg:
                curses.init_pair(2, curses.COLOR_RED, -1)
                curses.init_pair(3, curses.COLOR_GREEN, -1)
                curses.init_pair(5, curses.COLOR_MAGENTA, -1)
            else:
                curses.init_pair(2, curses.COLOR_WHITE, curses.COLOR_RED)
                curses.init_pair(3, curses.COLOR_WHITE, curses.COLOR_GREEN)
                curses.init_pair(5, curses.COLOR_WHITE, curses.COLOR_MAGENTA)
            curses.init_pair(4, curses.COLOR_BLUE, -1)
            curses.init_pair(6, curses.COLOR_RED, -1)
            curses.init_pair(7, curses.COLOR_GREEN, -1)
            curses.init_pair(8, curses.COLOR_MAGENTA, -1)
            self.no_color = curses.color_pair(1)
            self.default_color = curses.color_pair(3) | A_BOLD
            self.nice_color = curses.color_pair(8)
            self.cpu_time_color = curses.color_pair(8)
            self.ifCAREFUL_color = curses.color_pair(4) | A_BOLD
            self.ifWARNING_color = curses.color_pair(5) | A_BOLD
            self.ifCRITICAL_color = curses.color_pair(2) | A_BOLD
            self.default_color2 = curses.color_pair(7)
            self.ifCAREFUL_color2 = curses.color_pair(4)
            self.ifWARNING_color2 = curses.color_pair(8) | A_BOLD
            self.ifCRITICAL_color2 = curses.color_pair(6) | A_BOLD
            self.ifINFO_color = curses.color_pair(4)
            self.filter_color = A_BOLD
            self.selected_color = A_BOLD
            self.separator = curses.color_pair(1)
            if curses.COLORS > 8:
                colors_list = [curses.COLOR_CYAN, curses.COLOR_YELLOW]
                for i in range(0, 3):
                    try:
                        curses.init_pair(i + 9, colors_list[i], -1)
                    except Exception:
                        if self.is_theme('white'):
                            curses.init_pair(i + 9, curses.COLOR_BLACK, -1)
                        else:
                            curses.init_pair(i + 9, curses.COLOR_WHITE, -1)
                self.filter_color = curses.color_pair(9) | A_BOLD
                self.selected_color = curses.color_pair(10) | A_BOLD
                curses.init_color(11, 500, 500, 500)
                curses.init_pair(11, curses.COLOR_BLACK, -1)
                self.separator = curses.color_pair(11)
        else:
            self.no_color = curses.A_NORMAL
            self.default_color = curses.A_NORMAL
            self.nice_color = A_BOLD
            self.cpu_time_color = A_BOLD
            self.ifCAREFUL_color = A_BOLD
            self.ifWARNING_color = curses.A_UNDERLINE
            self.ifCRITICAL_color = curses.A_REVERSE
            self.default_color2 = curses.A_NORMAL
            self.ifCAREFUL_color2 = A_BOLD
            self.ifWARNING_color2 = curses.A_UNDERLINE
            self.ifCRITICAL_color2 = curses.A_REVERSE
            self.ifINFO_color = A_BOLD
            self.filter_color = A_BOLD
            self.selected_color = A_BOLD
            self.separator = curses.COLOR_BLACK
        self.colors_list = {'DEFAULT': self.no_color, 'UNDERLINE': curses.A_UNDERLINE, 'BOLD': A_BOLD, 'SORT': curses.A_UNDERLINE | A_BOLD, 'OK': self.default_color2, 'MAX': self.default_color2 | A_BOLD, 'FILTER': self.filter_color, 'TITLE': self.title_color, 'PROCESS': self.default_color2, 'PROCESS_SELECTED': self.default_color2 | curses.A_UNDERLINE, 'STATUS': self.default_color2, 'NICE': self.nice_color, 'CPU_TIME': self.cpu_time_color, 'CAREFUL': self.ifCAREFUL_color2, 'WARNING': self.ifWARNING_color2, 'CRITICAL': self.ifCRITICAL_color2, 'OK_LOG': self.default_color, 'CAREFUL_LOG': self.ifCAREFUL_color, 'WARNING_LOG': self.ifWARNING_color, 'CRITICAL_LOG': self.ifCRITICAL_color, 'PASSWORD': curses.A_PROTECT, 'SELECTED': self.selected_color, 'INFO': self.ifINFO_color, 'ERROR': self.selected_color, 'SEPARATOR': self.separator}

    def set_cursor(self, value):
        if False:
            print('Hello World!')
        'Configure the curse cursor appearance.\n\n        0: invisible\n        1: visible\n        2: very visible\n        '
        if hasattr(curses, 'curs_set'):
            try:
                curses.curs_set(value)
            except Exception:
                pass

    def get_key(self, window):
        if False:
            print('Hello World!')
        ret = window.getch()
        return ret

    def __catch_key(self, return_to_browser=False):
        if False:
            while True:
                i = 10
        self.pressedkey = self.get_key(self.term_window)
        if self.pressedkey == -1:
            return -1
        logger.debug('Keypressed (code: {})'.format(self.pressedkey))
        for hotkey in self._hotkeys:
            if self.pressedkey == ord(hotkey) and 'switch' in self._hotkeys[hotkey]:
                self._handle_switch(hotkey)
            elif self.pressedkey == ord(hotkey) and 'sort_key' in self._hotkeys[hotkey]:
                self._handle_sort_key(hotkey)
        if self.pressedkey == ord('\n'):
            self._handle_enter()
        elif self.pressedkey == ord('4'):
            self._handle_quicklook()
        elif self.pressedkey == ord('5'):
            self._handle_top_menu()
        elif self.pressedkey == ord('9'):
            self._handle_theme()
        elif self.pressedkey == ord('e') and (not self.args.programs):
            self._handle_process_extended()
        elif self.pressedkey == ord('E'):
            self._handle_erase_filter()
        elif self.pressedkey == ord('f'):
            self._handle_fs_stats()
        elif self.pressedkey == ord('+'):
            self._handle_increase_nice()
        elif self.pressedkey == ord('-'):
            self._handle_decrease_nice()
        elif self.pressedkey == ord('k') and (not self.args.disable_cursor):
            self._handle_kill_process()
        elif self.pressedkey == ord('w'):
            self._handle_clean_logs()
        elif self.pressedkey == ord('x'):
            self._handle_clean_critical_logs()
        elif self.pressedkey == ord('z'):
            self._handle_disable_process()
        elif self.pressedkey == curses.KEY_LEFT:
            self._handle_sort_left()
        elif self.pressedkey == curses.KEY_RIGHT:
            self._handle_sort_right()
        elif self.pressedkey == curses.KEY_UP or (self.pressedkey == 65 and (not self.args.disable_cursor)):
            self._handle_cursor_up()
        elif self.pressedkey == curses.KEY_DOWN or (self.pressedkey == 66 and (not self.args.disable_cursor)):
            self._handle_cursor_down()
        elif self.pressedkey == ord('\x1b') or self.pressedkey == ord('q'):
            self._handle_quit(return_to_browser)
        elif self.pressedkey == curses.KEY_F5 or self.pressedkey == 18:
            self._handle_refresh()
        return self.pressedkey

    def _handle_switch(self, hotkey):
        if False:
            i = 10
            return i + 15
        option = '_'.join(self._hotkeys[hotkey]['switch'].split('_')[1:])
        if self._hotkeys[hotkey]['switch'].startswith('disable_'):
            if getattr(self.args, self._hotkeys[hotkey]['switch']):
                enable(self.args, option)
            else:
                disable(self.args, option)
        elif self._hotkeys[hotkey]['switch'].startswith('enable_'):
            if getattr(self.args, self._hotkeys[hotkey]['switch']):
                disable(self.args, option)
            else:
                enable(self.args, option)
        else:
            setattr(self.args, self._hotkeys[hotkey]['switch'], not getattr(self.args, self._hotkeys[hotkey]['switch']))

    def _handle_sort_key(self, hotkey):
        if False:
            return 10
        glances_processes.set_sort_key(self._hotkeys[hotkey]['sort_key'], self._hotkeys[hotkey]['sort_key'] == 'auto')

    def _handle_enter(self):
        if False:
            while True:
                i = 10
        self.edit_filter = not self.edit_filter

    def _handle_quicklook(self):
        if False:
            while True:
                i = 10
        self.args.full_quicklook = not self.args.full_quicklook
        if self.args.full_quicklook:
            self.enable_fullquicklook()
        else:
            self.disable_fullquicklook()

    def _handle_top_menu(self):
        if False:
            i = 10
            return i + 15
        self.args.disable_top = not self.args.disable_top
        if self.args.disable_top:
            self.disable_top()
        else:
            self.enable_top()

    def _handle_theme(self):
        if False:
            i = 10
            return i + 15
        self._init_colors()

    def _handle_process_extended(self):
        if False:
            print('Hello World!')
        self.args.enable_process_extended = not self.args.enable_process_extended
        if not self.args.enable_process_extended:
            glances_processes.disable_extended()
        else:
            glances_processes.enable_extended()
        self.args.disable_cursor = self.args.enable_process_extended and self.args.is_standalone

    def _handle_erase_filter(self):
        if False:
            return 10
        glances_processes.process_filter = None

    def _handle_fs_stats(self):
        if False:
            i = 10
            return i + 15
        self.args.disable_fs = not self.args.disable_fs
        self.args.disable_folders = not self.args.disable_folders

    def _handle_increase_nice(self):
        if False:
            print('Hello World!')
        self.increase_nice_process = not self.increase_nice_process

    def _handle_decrease_nice(self):
        if False:
            print('Hello World!')
        self.decrease_nice_process = not self.decrease_nice_process

    def _handle_kill_process(self):
        if False:
            while True:
                i = 10
        self.kill_process = not self.kill_process

    def _handle_clean_logs(self):
        if False:
            for i in range(10):
                print('nop')
        glances_events.clean()

    def _handle_clean_critical_logs(self):
        if False:
            print('Hello World!')
        glances_events.clean(critical=True)

    def _handle_disable_process(self):
        if False:
            while True:
                i = 10
        self.args.disable_process = not self.args.disable_process
        if self.args.disable_process:
            glances_processes.disable()
        else:
            glances_processes.enable()

    def _handle_sort_left(self):
        if False:
            while True:
                i = 10
        next_sort = (self.loop_position() - 1) % len(self._sort_loop)
        glances_processes.set_sort_key(self._sort_loop[next_sort], False)

    def _handle_sort_right(self):
        if False:
            i = 10
            return i + 15
        next_sort = (self.loop_position() + 1) % len(self._sort_loop)
        glances_processes.set_sort_key(self._sort_loop[next_sort], False)

    def _handle_cursor_up(self):
        if False:
            i = 10
            return i + 15
        if self.args.cursor_position > 0:
            self.args.cursor_position -= 1

    def _handle_cursor_down(self):
        if False:
            while True:
                i = 10
        if self.args.cursor_position < glances_processes.processes_count:
            self.args.cursor_position += 1

    def _handle_quit(self, return_to_browser):
        if False:
            i = 10
            return i + 15
        if return_to_browser:
            logger.info('Stop Glances client and return to the browser')
        else:
            logger.info('Stop Glances (keypressed: {})'.format(self.pressedkey))

    def _handle_refresh(self):
        if False:
            while True:
                i = 10
        pass

    def loop_position(self):
        if False:
            for i in range(10):
                print('nop')
        'Return the current sort in the loop'
        for (i, v) in enumerate(self._sort_loop):
            if v == glances_processes.sort_key:
                return i
        return 0

    def disable_top(self):
        if False:
            for i in range(10):
                print('nop')
        'Disable the top panel'
        for p in ['quicklook', 'cpu', 'gpu', 'mem', 'memswap', 'load']:
            setattr(self.args, 'disable_' + p, True)

    def enable_top(self):
        if False:
            for i in range(10):
                print('nop')
        'Enable the top panel'
        for p in ['quicklook', 'cpu', 'gpu', 'mem', 'memswap', 'load']:
            setattr(self.args, 'disable_' + p, False)

    def disable_fullquicklook(self):
        if False:
            i = 10
            return i + 15
        'Disable the full quicklook mode'
        for p in ['quicklook', 'cpu', 'gpu', 'mem', 'memswap']:
            setattr(self.args, 'disable_' + p, False)

    def enable_fullquicklook(self):
        if False:
            while True:
                i = 10
        'Disable the full quicklook mode'
        self.args.disable_quicklook = False
        for p in ['cpu', 'gpu', 'mem', 'memswap']:
            setattr(self.args, 'disable_' + p, True)

    def end(self):
        if False:
            while True:
                i = 10
        'Shutdown the curses window.'
        if hasattr(curses, 'echo'):
            curses.echo()
        if hasattr(curses, 'nocbreak'):
            curses.nocbreak()
        if hasattr(curses, 'curs_set'):
            try:
                curses.curs_set(1)
            except Exception:
                pass
        curses.endwin()

    def init_line_column(self):
        if False:
            while True:
                i = 10
        'Init the line and column position for the curses interface.'
        self.init_line()
        self.init_column()

    def init_line(self):
        if False:
            i = 10
            return i + 15
        'Init the line position for the curses interface.'
        self.line = 0
        self.next_line = 0

    def init_column(self):
        if False:
            return 10
        'Init the column position for the curses interface.'
        self.column = 0
        self.next_column = 0

    def new_line(self, separator=False):
        if False:
            for i in range(10):
                print('nop')
        'New line in the curses interface.'
        self.line = self.next_line

    def new_column(self):
        if False:
            for i in range(10):
                print('nop')
        'New column in the curses interface.'
        self.column = self.next_column

    def separator_line(self, color='SEPARATOR'):
        if False:
            print('Hello World!')
        'New separator line in the curses interface.'
        if not self.args.enable_separator:
            return
        self.new_line()
        self.line -= 1
        line_width = self.term_window.getmaxyx()[1] - self.column
        self.term_window.addnstr(self.line, self.column, unicode_message('MEDIUM_LINE', self.args) * line_width, line_width, self.colors_list[color])

    def __get_stat_display(self, stats, layer):
        if False:
            print('Hello World!')
        'Return a dict of dict with all the stats display.\n        # TODO: Drop extra parameter\n\n        :param stats: Global stats dict\n        :param layer: ~ cs_status\n            "None": standalone or server mode\n            "Connected": Client is connected to a Glances server\n            "SNMP": Client is connected to a SNMP server\n            "Disconnected": Client is disconnected from the server\n\n        :returns: dict of dict\n            * key: plugin name\n            * value: dict returned by the get_stats_display Plugin method\n        '
        ret = {}
        for p in stats.getPluginsList(enable=False):
            if p == 'quicklook' or p == 'processlist':
                continue
            plugin_max_width = None
            if p in self._left_sidebar:
                plugin_max_width = max(self._left_sidebar_min_width, self.term_window.getmaxyx()[1] - 105)
                plugin_max_width = min(self._left_sidebar_max_width, plugin_max_width)
            ret[p] = stats.get_plugin(p).get_stats_display(args=self.args, max_width=plugin_max_width)
        return ret

    def display(self, stats, cs_status=None):
        if False:
            print('Hello World!')
        'Display stats on the screen.\n\n        :param stats: Stats database to display\n        :param cs_status:\n            "None": standalone or server mode\n            "Connected": Client is connected to a Glances server\n            "SNMP": Client is connected to a SNMP server\n            "Disconnected": Client is disconnected from the server\n\n        :return: True if the stats have been displayed else False if the help have been displayed\n        '
        self.init_line_column()
        self.args.cs_status = cs_status
        __stat_display = self.__get_stat_display(stats, layer=cs_status)
        max_processes_displayed = self.term_window.getmaxyx()[0] - 11 - (0 if 'containers' not in __stat_display else self.get_stats_display_height(__stat_display['containers'])) - (0 if 'processcount' not in __stat_display else self.get_stats_display_height(__stat_display['processcount'])) - (0 if 'amps' not in __stat_display else self.get_stats_display_height(__stat_display['amps'])) - (0 if 'alert' not in __stat_display else self.get_stats_display_height(__stat_display['alert']))
        try:
            if self.args.enable_process_extended:
                max_processes_displayed -= 4
        except AttributeError:
            pass
        if max_processes_displayed < 0:
            max_processes_displayed = 0
        if glances_processes.max_processes is None or glances_processes.max_processes != max_processes_displayed:
            logger.debug('Set number of displayed processes to {}'.format(max_processes_displayed))
            glances_processes.max_processes = max_processes_displayed
        __stat_display['processlist'] = stats.get_plugin('processlist').get_stats_display(args=self.args)
        if self.args.help_tag:
            self.display_plugin(stats.get_plugin('help').get_stats_display(args=self.args))
            return False
        self.__display_header(__stat_display)
        self.separator_line()
        self.__display_top(__stat_display, stats)
        self.init_column()
        self.separator_line()
        self.__display_left(__stat_display)
        self.__display_right(__stat_display)
        if self.edit_filter and cs_status is None:
            new_filter = self.display_popup('Process filter pattern: \n\n' + 'Examples:\n' + '- .*python.*\n' + '- /usr/lib.*\n' + '- name:.*nautilus.*\n' + '- cmdline:.*glances.*\n' + '- username:nicolargo\n' + '- username:^root        ', popup_type='input', input_value=glances_processes.process_filter_input)
            glances_processes.process_filter = new_filter
        elif self.edit_filter and cs_status is not None:
            self.display_popup('Process filter only available in standalone mode')
        self.edit_filter = False
        if self.increase_nice_process and cs_status is None:
            self.nice_increase(stats.get_plugin('processlist').get_raw()[self.args.cursor_position])
        self.increase_nice_process = False
        if self.decrease_nice_process and cs_status is None:
            self.nice_decrease(stats.get_plugin('processlist').get_raw()[self.args.cursor_position])
        self.decrease_nice_process = False
        if self.kill_process and cs_status is None:
            self.kill(stats.get_plugin('processlist').get_raw()[self.args.cursor_position])
        elif self.kill_process and cs_status is not None:
            self.display_popup('Kill process only available for local processes')
        self.kill_process = False
        if self.args.generate_graph:
            if 'graph' in stats.getExportsList():
                self.display_popup('Generate graph in {}'.format(self.args.export_graph_path))
            else:
                logger.warning('Graph export module is disable. Run Glances with --export graph to enable it.')
                self.args.generate_graph = False
        return True

    def nice_increase(self, process):
        if False:
            while True:
                i = 10
        glances_processes.nice_increase(process['pid'])

    def nice_decrease(self, process):
        if False:
            return 10
        glances_processes.nice_decrease(process['pid'])

    def kill(self, process):
        if False:
            return 10
        'Kill a process, or a list of process if the process has a childrens field.\n\n        :param process\n        :return: None\n        '
        logger.debug('Selected process to kill: {}'.format(process))
        if 'childrens' in process:
            pid_to_kill = process['childrens']
        else:
            pid_to_kill = [process['pid']]
        confirm = self.display_popup('Kill process: {} (pid: {}) ?\n\nConfirm ([y]es/[n]o): '.format(process['name'], ', '.join(map(str, pid_to_kill))), popup_type='yesno')
        if confirm.lower().startswith('y'):
            for pid in pid_to_kill:
                try:
                    ret_kill = glances_processes.kill(pid)
                except Exception as e:
                    logger.error('Can not kill process {} ({})'.format(pid, e))
                else:
                    logger.info('Kill signal has been sent to process {} (return code: {})'.format(pid, ret_kill))

    def __display_header(self, stat_display):
        if False:
            for i in range(10):
                print('nop')
        'Display the firsts lines (header) in the Curses interface.\n\n        system + ip + uptime\n        (cloud)\n        '
        self.new_line()
        self.space_between_column = 0
        l_uptime = 1
        for i in ['system', 'ip', 'uptime']:
            if i in stat_display:
                l_uptime += self.get_stats_display_width(stat_display[i])
        self.display_plugin(stat_display['system'], display_optional=self.term_window.getmaxyx()[1] >= l_uptime)
        self.space_between_column = 3
        if 'ip' in stat_display:
            self.new_column()
            self.display_plugin(stat_display['ip'], display_optional=self.term_window.getmaxyx()[1] >= 100)
        self.new_column()
        self.display_plugin(stat_display['uptime'], add_space=-(self.get_stats_display_width(stat_display['cloud']) != 0))
        self.init_column()
        if self.get_stats_display_width(stat_display['cloud']) != 0:
            self.new_line()
            self.display_plugin(stat_display['cloud'])

    def __display_top(self, stat_display, stats):
        if False:
            print('Hello World!')
        'Display the second line in the Curses interface.\n\n        <QUICKLOOK> + CPU|PERCPU + <GPU> + MEM + SWAP + LOAD\n        '
        self.init_column()
        self.new_line()
        stat_display['quicklook'] = {'msgdict': []}
        plugin_widths = {}
        for p in self._top:
            plugin_widths[p] = self.get_stats_display_width(stat_display.get(p, 0)) if hasattr(self.args, 'disable_' + p) else 0
        stats_width = sum(itervalues(plugin_widths))
        stats_number = sum([int(stat_display[p]['msgdict'] != []) for p in self._top if not getattr(self.args, 'disable_' + p)])
        if not self.args.disable_quicklook:
            if self.args.full_quicklook:
                quicklook_width = self.term_window.getmaxyx()[1] - (stats_width + 8 + stats_number * self.space_between_column)
            else:
                quicklook_width = min(self.term_window.getmaxyx()[1] - (stats_width + 8 + stats_number * self.space_between_column), self._quicklook_max_width - 5)
            try:
                stat_display['quicklook'] = stats.get_plugin('quicklook').get_stats_display(max_width=quicklook_width, args=self.args)
            except AttributeError as e:
                logger.debug('Quicklook plugin not available (%s)' % e)
            else:
                plugin_widths['quicklook'] = self.get_stats_display_width(stat_display['quicklook'])
                stats_width = sum(itervalues(plugin_widths)) + 1
            self.space_between_column = 1
            self.display_plugin(stat_display['quicklook'])
            self.new_column()
        plugin_display_optional = {}
        for p in self._top:
            plugin_display_optional[p] = True
        if stats_number > 1:
            self.space_between_column = max(1, int((self.term_window.getmaxyx()[1] - stats_width) / (stats_number - 1)))
            for p in ['mem', 'cpu']:
                if self.space_between_column < 3:
                    plugin_display_optional[p] = False
                    plugin_widths[p] = self.get_stats_display_width(stat_display[p], without_option=True) if hasattr(self.args, 'disable_' + p) else 0
                    stats_width = sum(itervalues(plugin_widths)) + 1
                    self.space_between_column = max(1, int((self.term_window.getmaxyx()[1] - stats_width) / (stats_number - 1)))
        else:
            self.space_between_column = 0
        for p in self._top:
            if p == 'quicklook':
                continue
            if p in stat_display:
                self.display_plugin(stat_display[p], display_optional=plugin_display_optional[p])
            if p != 'load':
                self.new_column()
        self.space_between_column = 3
        self.saved_line = self.next_line

    def __display_left(self, stat_display):
        if False:
            print('Hello World!')
        'Display the left sidebar in the Curses interface.'
        self.init_column()
        if self.args.disable_left_sidebar:
            return
        for p in self._left_sidebar:
            if (hasattr(self.args, 'enable_' + p) or hasattr(self.args, 'disable_' + p)) and p in stat_display:
                self.new_line()
                self.display_plugin(stat_display[p])

    def __display_right(self, stat_display):
        if False:
            print('Hello World!')
        'Display the right sidebar in the Curses interface.\n\n        docker + processcount + amps + processlist + alert\n        '
        if self.term_window.getmaxyx()[1] < self._left_sidebar_min_width:
            return
        self.next_line = self.saved_line
        self.new_column()
        for p in self._right_sidebar:
            if (hasattr(self.args, 'enable_' + p) or hasattr(self.args, 'disable_' + p)) and p in stat_display:
                if p not in p:
                    continue
                self.new_line()
                if p == 'processlist':
                    self.display_plugin(stat_display['processlist'], display_optional=self.term_window.getmaxyx()[1] > 102, display_additional=not MACOS, max_y=self.term_window.getmaxyx()[0] - self.get_stats_display_height(stat_display['alert']) - 2)
                else:
                    self.display_plugin(stat_display[p])

    def display_popup(self, message, size_x=None, size_y=None, duration=3, popup_type='info', input_size=30, input_value=None):
        if False:
            i = 10
            return i + 15
        "\n        Display a centered popup.\n\n         popup_type: ='info'\n         Just an information popup, no user interaction\n         Display a centered popup with the given message during duration seconds\n         If size_x and size_y: set the popup size\n         else set it automatically\n         Return True if the popup could be displayed\n\n        popup_type='input'\n         Display a centered popup with the given message and a input field\n         If size_x and size_y: set the popup size\n         else set it automatically\n         Return the input string or None if the field is empty\n\n        popup_type='yesno'\n         Display a centered popup with the given message\n         If size_x and size_y: set the popup size\n         else set it automatically\n         Return True (yes) or False (no)\n        "
        sentence_list = message.split('\n')
        if size_x is None:
            size_x = len(max(sentence_list, key=len)) + 4
            if popup_type == 'input':
                size_x += input_size
        if size_y is None:
            size_y = len(sentence_list) + 4
        screen_x = self.term_window.getmaxyx()[1]
        screen_y = self.term_window.getmaxyx()[0]
        if size_x > screen_x or size_y > screen_y:
            return False
        pos_x = int((screen_x - size_x) / 2)
        pos_y = int((screen_y - size_y) / 2)
        popup = curses.newwin(size_y, size_x, pos_y, pos_x)
        popup.border()
        for (y, m) in enumerate(sentence_list):
            popup.addnstr(2 + y, 2, m, len(m))
        if popup_type == 'info':
            popup.refresh()
            self.wait(duration * 1000)
            return True
        elif popup_type == 'input':
            sub_pop = popup.derwin(1, input_size, 2, 2 + len(m))
            sub_pop.attron(self.colors_list['FILTER'])
            if input_value is not None:
                sub_pop.addnstr(0, 0, input_value, len(input_value))
            popup.refresh()
            sub_pop.refresh()
            self.set_cursor(2)
            self.term_window.keypad(1)
            textbox = GlancesTextbox(sub_pop, insert_mode=True)
            textbox.edit()
            self.set_cursor(0)
            if textbox.gather() != '':
                logger.debug('User enters the following string: %s' % textbox.gather())
                return textbox.gather()[:-1]
            else:
                logger.debug('User centers an empty string')
                return None
        elif popup_type == 'yesno':
            sub_pop = popup.derwin(1, 2, len(sentence_list) + 1, len(m) + 2)
            sub_pop.attron(self.colors_list['FILTER'])
            sub_pop.addnstr(0, 0, '', 0)
            popup.refresh()
            sub_pop.refresh()
            self.set_cursor(2)
            self.term_window.keypad(1)
            textbox = GlancesTextboxYesNo(sub_pop, insert_mode=False)
            textbox.edit()
            self.set_cursor(0)
            return textbox.gather()

    def display_plugin(self, plugin_stats, display_optional=True, display_additional=True, max_y=65535, add_space=0):
        if False:
            print('Hello World!')
        'Display the plugin_stats on the screen.\n\n        :param plugin_stats:\n        :param display_optional: display the optional stats if True\n        :param display_additional: display additional stats if True\n        :param max_y: do not display line > max_y\n        :param add_space: add x space (line) after the plugin\n        '
        if plugin_stats is None or not plugin_stats['msgdict'] or (not plugin_stats['display']):
            return 0
        screen_x = self.term_window.getmaxyx()[1]
        screen_y = self.term_window.getmaxyx()[0]
        if plugin_stats['align'] == 'right':
            display_x = screen_x - self.get_stats_display_width(plugin_stats)
        else:
            display_x = self.column
        if plugin_stats['align'] == 'bottom':
            display_y = screen_y - self.get_stats_display_height(plugin_stats)
        else:
            display_y = self.line
        x = display_x
        x_max = x
        y = display_y
        for m in plugin_stats['msgdict']:
            try:
                if m['msg'].startswith('\n'):
                    y += 1
                    x = display_x
                    continue
            except Exception:
                pass
            if x < 0:
                continue
            if not m['splittable'] and x + len(m['msg']) > screen_x:
                continue
            if y < 0 or y + 1 > screen_y or y > max_y:
                break
            if not display_optional and m['optional']:
                continue
            if not display_additional and m['additional']:
                continue
            try:
                self.term_window.addnstr(y, x, m['msg'], screen_x - x, self.colors_list[m['decoration']])
            except Exception:
                pass
            else:
                try:
                    x += len(u(m['msg']))
                except UnicodeDecodeError:
                    pass
                if x > x_max:
                    x_max = x
        self.next_column = max(self.next_column, x_max + self.space_between_column)
        self.next_line = max(self.next_line, y + self.space_between_line)
        self.next_line += add_space

    def clear(self):
        if False:
            print('Hello World!')
        'Erase the content of the screen.\n        The difference is that clear() also calls clearok(). clearok()\n        basically tells ncurses to forget whatever it knows about the current\n        terminal contents, so that when refresh() is called, it will actually\n        begin by clearing the entire terminal screen before redrawing any of it.'
        self.term_window.clear()

    def erase(self):
        if False:
            print('Hello World!')
        'Erase the content of the screen.\n        erase() on the other hand, just clears the screen (the internal\n        object, not the terminal screen). When refresh() is later called,\n        ncurses will still compute the minimum number of characters to send to\n        update the terminal.'
        self.term_window.erase()

    def flush(self, stats, cs_status=None):
        if False:
            for i in range(10):
                print('nop')
        'Erase and update the screen.\n\n        :param stats: Stats database to display\n        :param cs_status:\n            "None": standalone or server mode\n            "Connected": Client is connected to the server\n            "Disconnected": Client is disconnected from the server\n        '
        self.erase()
        self.display(stats, cs_status=cs_status)

    def update(self, stats, duration=3, cs_status=None, return_to_browser=False):
        if False:
            while True:
                i = 10
        'Update the screen.\n\n        :param stats: Stats database to display\n        :param duration: duration of the loop\n        :param cs_status:\n            "None": standalone or server mode\n            "Connected": Client is connected to the server\n            "Disconnected": Client is disconnected from the server\n        :param return_to_browser:\n            True: Do not exist, return to the browser list\n            False: Exit and return to the shell\n\n        :return: True if exit key has been pressed else False\n        '
        self.flush(stats, cs_status=cs_status)
        if duration <= 0:
            logger.warning('Update and export time higher than refresh_time.')
            duration = 0.1
        isexitkey = False
        countdown = Timer(duration)
        self.term_window.timeout(100)
        while not countdown.finished() and (not isexitkey):
            pressedkey = self.__catch_key(return_to_browser=return_to_browser)
            isexitkey = pressedkey == ord('\x1b') or pressedkey == ord('q')
            if pressedkey == curses.KEY_F5 or self.pressedkey == 18:
                self.clear()
                return isexitkey
            if pressedkey in (curses.KEY_UP, 65, curses.KEY_DOWN, 66):
                countdown.reset()
            if isexitkey and self.args.help_tag:
                self.args.help_tag = not self.args.help_tag
                isexitkey = False
                return isexitkey
            if not isexitkey and pressedkey > -1:
                self.flush(stats, cs_status=cs_status)
                self.wait(delay=int(countdown.get() * 1000))
        return isexitkey

    def wait(self, delay=100):
        if False:
            return 10
        'Wait delay in ms'
        curses.napms(delay)

    def get_stats_display_width(self, curse_msg, without_option=False):
        if False:
            for i in range(10):
                print('nop')
        'Return the width of the formatted curses message.'
        try:
            if without_option:
                c = len(max(''.join([u(u(nativestr(i['msg'])).encode('ascii', 'replace')) if not i['optional'] else '' for i in curse_msg['msgdict']]).split('\n'), key=len))
            else:
                c = len(max(''.join([u(u(nativestr(i['msg'])).encode('ascii', 'replace')) for i in curse_msg['msgdict']]).split('\n'), key=len))
        except Exception as e:
            logger.debug('ERROR: Can not compute plugin width ({})'.format(e))
            return 0
        else:
            return c

    def get_stats_display_height(self, curse_msg):
        if False:
            while True:
                i = 10
        "Return the height of the formatted curses message.\n\n        The height is defined by the number of '\n' (new line).\n        "
        try:
            c = [i['msg'] for i in curse_msg['msgdict']].count('\n')
        except Exception as e:
            logger.debug('ERROR: Can not compute plugin height ({})'.format(e))
            return 0
        else:
            return c + 1

class GlancesCursesStandalone(_GlancesCurses):
    """Class for the Glances curse standalone."""

class GlancesCursesClient(_GlancesCurses):
    """Class for the Glances curse client."""

class GlancesTextbox(Textbox, object):

    def __init__(self, *args, **kwargs):
        if False:
            return 10
        super(GlancesTextbox, self).__init__(*args, **kwargs)

    def do_command(self, ch):
        if False:
            while True:
                i = 10
        if ch == 10:
            return 0
        if ch == 127:
            return 8
        return super(GlancesTextbox, self).do_command(ch)

class GlancesTextboxYesNo(Textbox, object):

    def __init__(self, *args, **kwargs):
        if False:
            return 10
        super(GlancesTextboxYesNo, self).__init__(*args, **kwargs)

    def do_command(self, ch):
        if False:
            i = 10
            return i + 15
        return super(GlancesTextboxYesNo, self).do_command(ch)