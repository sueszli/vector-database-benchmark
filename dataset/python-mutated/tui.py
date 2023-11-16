"""
This module was made to handle the curses sections for the ap selection,
template selection and the main window
"""
import curses
import os
import re
import time
from collections import namedtuple
from subprocess import check_output
import wifiphisher.common.accesspoint as accesspoint
import wifiphisher.common.constants as constants
import wifiphisher.common.phishingpage as phishingpage
import wifiphisher.common.recon as recon
import wifiphisher.common.victim as victim
MainInfo = namedtuple('MainInfo', constants.MAIN_TUI_ATTRS)
ApSelInfo = namedtuple('ApSelInfo', constants.AP_SEL_ATTRS)

class TuiTemplateSelection(object):
    """
    TUI to do Template selection
    """

    def __init__(self):
        if False:
            print('Hello World!')
        '\n        Construct the class\n        :param self: A TuiTemplateSelection object\n        :type self: TuiTemplateSelection\n        :return None\n        :rtype None\n        '
        self.green_text = None
        self.heightlight_text = None
        self.heightlight_number = 0
        self.page_number = 0
        self.sections = list()
        self.sec_page_map = {}
        self.dimension = [0, 0]

    def get_sections(self, template_names, templates):
        if False:
            while True:
                i = 10
        '\n        Get all the phishing scenario contents and store them\n        in a list\n        :param self: A TuiTemplateSelection object\n        :param template_names: A list of string\n        :param templates: A dictionary\n        :type self: TuiTemplateSelection\n        :type template_names: list\n        :type templates: dict\n        :return None\n        :rtype: None\n        '
        for name in template_names:
            phishing_contents = ' - ' + str(templates[name])
            lines = phishing_contents.splitlines()
            short_lines = []
            for line in lines:
                for short_line in line_splitter(15, line):
                    short_lines.append(short_line)
            self.sections.append(short_lines)

    def update_sec_page_map(self, last_row):
        if False:
            for i in range(10):
                print('nop')
        '\n        Update the page number for each section\n        :param self: A TuiTemplateSelection object\n        :param last_row: The last row of the window\n        :type self: TuiTemplateSelection\n        :type last_row: int\n        :return: None\n        :rtype: None\n        '
        page_number = 0
        row_number = 0
        self.sec_page_map = {}
        for (number, section) in enumerate(self.sections):
            row_number += len(section)
            if row_number > last_row:
                row_number = 0
                page_number += 1
            self.sec_page_map[number] = page_number

    def gather_info(self, template_argument, template_manager):
        if False:
            i = 10
            return i + 15
        '\n        Select a template based on whether the template argument\n        is set or not. If the template argument is not set, it will\n        interfactively ask user for a template\n        :param self: A TuiTemplateSelection object\n        :type self: TuiTemplateSelection\n        :param template_argument: The template argument which might\n        have been entered by the user\n        :type template_argument: str\n        :param template_manager: A TemplateManager object\n        :type template_manager: TemplateManager\n        :return A PhishingTemplate object\n        :rtype: PhishingTemplagte\n        :raises  InvalidTemplate in case the template argument entered\n        by the user is not available.\n        '
        templates = template_manager.get_templates()
        template_names = list(templates.keys())
        self.get_sections(template_names, templates)
        if template_argument and template_argument in templates:
            return templates[template_argument]
        elif template_argument and template_argument not in templates:
            raise phishingpage.InvalidTemplate
        else:
            template = curses.wrapper(self.display_info, templates, template_names)
        return template

    def key_movement(self, screen, number_of_sections, key):
        if False:
            print('Hello World!')
        '\n        Check for key movement and hightlight the corresponding\n        phishing scenario\n\n        :param self: A TuiTemplateSelection object\n        :param number_of_sections: Number of templates\n        :param key: The char user keying\n        :type self: TuiTemplateSelection\n        :type number_of_sections: int\n        :type key: str\n        :return: None\n        :rtype: None\n        '
        if key == curses.KEY_DOWN:
            if self.heightlight_number < number_of_sections - 1:
                page_number = self.sec_page_map[self.heightlight_number + 1]
                if page_number > self.page_number:
                    self.page_number += 1
                    screen.erase()
                self.heightlight_number += 1
        elif key == curses.KEY_UP:
            if self.heightlight_number > 0:
                page_number = self.sec_page_map[self.heightlight_number - 1]
                if page_number < self.page_number:
                    self.page_number -= 1
                    screen.erase()
                self.heightlight_number -= 1

    def display_phishing_scenarios(self, screen):
        if False:
            i = 10
            return i + 15
        '\n        Display the phishing scenarios\n        :param self: A TuiTemplateSelection object\n        :type self: TuiTemplateSelection\n        :param screen: A curses window object\n        :type screen: _curses.curses.window\n        :return total row numbers used to display the phishing scenarios\n        :rtype: int\n        '
        try:
            (max_window_height, max_window_len) = screen.getmaxyx()
            if self.dimension[0] != max_window_height or self.dimension[1] != max_window_len:
                screen.erase()
            self.dimension[0] = max_window_height
            self.dimension[1] = max_window_len
            self.update_sec_page_map(max_window_height - 20)
            display_str = 'Options: [Up Arrow] Move Up  [Down Arrow] Move Down'
            screen.addstr(0, 0, display_string(max_window_len, display_str))
            display_str = 'Available Phishing Scenarios:'
            screen.addstr(3, 0, display_string(max_window_len, display_str), curses.A_BOLD)
        except curses.error:
            return 0
        row_num = 5
        first = False
        for (number, short_lines) in enumerate(self.sections):
            try:
                if self.sec_page_map[self.heightlight_number] != self.page_number and (not first):
                    screen.addstr(row_num, 2, short_lines[0], self.heightlight_text)
                    self.heightlight_number = 0
                    self.page_number = 0
                    first = True
                if self.sec_page_map[number] != self.page_number:
                    continue
                screen.addstr(row_num, 0, str(number + 1), self.green_text)
                if number == self.heightlight_number:
                    screen.addstr(row_num, 2, short_lines[0], self.heightlight_text)
                else:
                    screen.addstr(row_num, 2, short_lines[0], curses.A_BOLD)
                row_num += 1
                screen.addstr(row_num, 8, short_lines[1])
                row_num += 1
                if len(short_lines) > 1:
                    for short_line in short_lines[2:]:
                        screen.addstr(row_num, 0, short_line)
                        row_num += 1
                row_num += 1
            except curses.error:
                return row_num
        return row_num

    def display_info(self, screen, templates, template_names):
        if False:
            print('Hello World!')
        '\n        Display the template information to users\n        :param self: A TuiTemplateSelection object\n        :type self: TuiTemplateSelection\n        :param screen: A curses window object\n        :type screen: _curses.curses.window\n        :param templates: A dictionay map page to PhishingTemplate\n        :type templates: dict\n        :param template_names: list of template names\n        :type template_names: list\n        '
        try:
            curses.curs_set(0)
        except curses.error:
            pass
        screen.nodelay(True)
        curses.init_pair(1, curses.COLOR_GREEN, screen.getbkgd())
        curses.init_pair(2, curses.COLOR_BLACK, curses.COLOR_CYAN)
        self.green_text = curses.color_pair(1) | curses.A_BOLD
        self.heightlight_text = curses.color_pair(2) | curses.A_BOLD
        number_of_sections = len(templates)
        screen.erase()
        while True:
            row_number = self.display_phishing_scenarios(screen)
            key = screen.getch()
            self.key_movement(screen, number_of_sections, key)
            row_number += 2
            if key == ord('\n'):
                try:
                    screen.addstr(row_number, 3, 'YOU HAVE SELECTED ' + template_names[self.heightlight_number], curses.A_BOLD)
                except curses.error:
                    pass
                screen.refresh()
                time.sleep(1)
                template_name = template_names[self.heightlight_number]
                template = templates[template_name]
                return template
            screen.refresh()

class ApDisplayInfo(object):
    """
    ApDisplayInfo class to store the information for ap selection
    """

    def __init__(self, pos, page_number, box, box_info):
        if False:
            return 10
        '\n        Construct the class\n        :param self: ApDisplayInfo\n        :param pos: position of the line in the ap selection page\n        :param page_number: page number of the ap selection\n        :param box: the curses.newwin.box object containing ap information\n        :param key: the key user have keyed in\n        :param box_info: list of window height, window len, and max row number\n        :type self: ApDisplayInfo\n        :type pos: int\n        :type page_number: int\n        :type box: curse.newwin.box\n        :type key: str\n        :return: None\n        :rtype: None\n        '
        self.pos = pos
        self.page_number = page_number
        self.box = box
        self._box_info = box_info

    @property
    def max_h(self):
        if False:
            return 10
        '\n        The height of the terminal screen\n        :param self: ApDisplayInfo\n        :type self: ApDisplayInfo\n        :return: the height of terminal screen\n        :rtype: int\n        '
        return self._box_info[0]

    @max_h.setter
    def max_h(self, val):
        if False:
            while True:
                i = 10
        '\n        Set the height of the terminal screen\n        :param self: ApDisplayInfo\n        :type self: ApDisplayInfo\n        :return: None\n        :rtype: None\n        '
        self._box_info[0] = val

    @property
    def max_l(self):
        if False:
            i = 10
            return i + 15
        '\n        The width of the terminal screen\n        :param self: ApDisplayInfo\n        :type self: ApDisplayInfo\n        :return: the width of terminal screen\n        :rtype: int\n        '
        return self._box_info[1]

    @max_l.setter
    def max_l(self, val):
        if False:
            while True:
                i = 10
        '\n        Set the width of the terminal screen\n        :param self: ApDisplayInfo\n        :type self: ApDisplayInfo\n        :return: None\n        :rtype: None\n        '
        self._box_info[1] = val

    @property
    def max_row(self):
        if False:
            while True:
                i = 10
        '\n        Maximum row numbers used to contain the ap information\n        :param self: ApDisplayInfo\n        :type self: ApDisplayInfo\n        :return: The row numbers of the box that contains the ap info\n        :rtype: int\n        '
        return self._box_info[2]

    @max_row.setter
    def max_row(self, val):
        if False:
            i = 10
            return i + 15
        '\n        Set maximum row numbers used to contain the ap information\n        :param self: ApDisplayInfo\n        :type self: ApDisplayInfo\n        :return: None\n        :rtype: None\n        '
        self._box_info[2] = val

    @property
    def key(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Get the key the users have keyed\n        :param self: ApDisplayInfo\n        :type self: ApDisplayInfo\n        :return: The key\n        :rtype: int\n        '
        return self._box_info[3]

    @key.setter
    def key(self, val):
        if False:
            i = 10
            return i + 15
        '\n        Set the key the users have keyed\n        :param self: ApDisplayInfo\n        :type self: ApDisplayInfo\n        :return: None\n        :rtype: None\n        '
        self._box_info[3] = val

class TuiApSel(object):
    """
    TuiApSel class to represent the ap selection terminal window
    """

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Construct the class\n        :param self: A TuiApSel object\n        :type self: TuiApSel\n        :return: None\n        :rtype: None\n        '
        self.total_ap_number = 0
        self.access_points = list()
        self.access_point_finder = None
        self.highlight_text = None
        self.normal_text = None
        self.mac_matcher = None
        self.renew_box = False

    def init_display_info(self, screen, info):
        if False:
            for i in range(10):
                print('nop')
        '\n        Initialization of the ApDisplyInfo object\n        :param self: A TuiApSel object\n        :type self: TuiApSel\n        :param screen: A curses window object\n        :type screen: _curses.curses.window\n        :param info: A namedtuple of information from pywifiphisher\n        :type info: namedtuple\n        :return ApDisplayInfo object\n        :rtype: ApDisplayInfo\n        '
        position = 1
        page_number = 1
        (max_window_height, max_window_length) = screen.getmaxyx()
        if max_window_height < 14 or max_window_length < 9:
            box = curses.newwin(max_window_height, max_window_length, 0, 0)
            self.renew_box = True
        else:
            box = curses.newwin(max_window_height - 9, max_window_length - 5, 4, 3)
        box.box()
        box_height = box.getmaxyx()[0]
        max_row = box_height - 2
        key = 0
        box_info = [max_window_height, max_window_length, max_row, key]
        ap_info = ApDisplayInfo(position, page_number, box, box_info)
        self.mac_matcher = info.mac_matcher
        self.access_point_finder = recon.AccessPointFinder(info.interface, info.network_manager)
        if info.args.lure10_capture:
            self.access_point_finder.capture_aps()
        self.access_point_finder.find_all_access_points()
        return ap_info

    def gather_info(self, screen, info):
        if False:
            while True:
                i = 10
        '\n        Get the information from pywifiphisher and print them out\n        :param self: A TuiApSel object\n        :type self: TuiApSel\n        :param screen: A curses window object\n        :type screen: _curses.curses.window\n        :param info: A namedtuple of information from pywifiphisher\n        :type info: namedtuple\n        :return AccessPoint object if users type enter\n        :rtype AccessPoint if users type enter else None\n        '
        try:
            curses.curs_set(0)
        except curses.error:
            pass
        screen.nodelay(True)
        curses.init_pair(1, curses.COLOR_BLACK, curses.COLOR_CYAN)
        self.highlight_text = curses.color_pair(1)
        self.normal_text = curses.A_NORMAL
        ap_info = self.init_display_info(screen, info)
        while ap_info.key != 27:
            is_done = self.display_info(screen, ap_info)
            if is_done:
                self.access_point_finder.stop_finding_access_points()
                return self.access_points[ap_info.pos - 1]
        self.access_point_finder.stop_finding_access_points()

    def resize_window(self, screen, ap_info):
        if False:
            for i in range(10):
                print('nop')
        '\n        Resize the window if the dimensions have been changed\n\n        :param self: A TuiApSel object\n        :type self: TuiApSel\n        :param screen: A curses window object\n        :type screen: _curses.curses.window\n        :param ap_info: An ApDisplayInfo object\n        :type ap_info: ApDisplayInfo\n        '
        if screen.getmaxyx() != (ap_info.max_h, ap_info.max_l):
            (ap_info.max_h, ap_info.max_l) = screen.getmaxyx()
            if ap_info.max_h < 10 + 4 or ap_info.max_l < 6 + 3:
                box = curses.newwin(ap_info.max_h, ap_info.max_l, 0, 0)
                box.box()
                ap_info.box = box
                self.renew_box = True
                return
            elif self.renew_box:
                screen.erase()
                box = curses.newwin(ap_info.max_h - 9, ap_info.max_l - 5, 4, 3)
                box.box()
                ap_info.box = box
                self.renew_box = False
            ap_info.box.resize(ap_info.max_h - 9, ap_info.max_l - 5)
            box_height = ap_info.box.getmaxyx()[0]
            ap_info.max_row = box_height - 2
            ap_info.pos = 1
            ap_info.page_number = 1

    def key_movement(self, ap_info):
        if False:
            return 10
        "\n        Check for any key movement and update it's result\n\n        :param self: A TuiApSel object\n        :type self: TuiApSel\n        :param ap_info: ApDisplayInfo object\n        :type: ApDisplayInfo\n        :return: None\n        :rtype: None\n        "
        key = ap_info.key
        pos = ap_info.pos
        max_row = ap_info.max_row
        page_number = ap_info.page_number
        if key == curses.KEY_DOWN:
            try:
                self.access_points[pos]
            except IndexError:
                ap_info.key = 0
                ap_info.pos = pos
                ap_info.max_row = max_row
                return
            if pos % max_row == 0:
                pos += 1
                page_number += 1
            else:
                pos += 1
        elif key == curses.KEY_UP:
            if pos - 1 > 0:
                if (pos - 1) % max_row == 0:
                    pos -= 1
                    page_number -= 1
                else:
                    pos -= 1
        ap_info.key = key
        ap_info.pos = pos
        ap_info.page_number = page_number

    def display_info(self, screen, ap_info):
        if False:
            i = 10
            return i + 15
        '\n        Display the AP informations on the screen\n\n        :param self: A TuiApSel object\n        :type self: TuiApSel\n        :param screen: A curses window object\n        :type screen: _curses.curses.window\n        :param ap_info: An ApDisplayInfo object\n        :type ap_info: ApDisplayInfo\n        :return True if ap selection is done\n        :rtype: bool\n        '
        is_apsel_end = False
        self.resize_window(screen, ap_info)
        new_total_ap_number = len(self.access_point_finder.observed_access_points)
        if new_total_ap_number != self.total_ap_number:
            self.access_points = self.access_point_finder.get_sorted_access_points()
            self.total_ap_number = len(self.access_points)
        self.display_access_points(screen, ap_info)
        self.key_movement(ap_info)
        ap_info.key = screen.getch()
        if ap_info.key == ord('\n') and self.total_ap_number != 0:
            screen.addstr(ap_info.max_h - 2, 3, 'YOU HAVE SELECTED ' + self.access_points[ap_info.pos - 1].name)
            screen.refresh()
            time.sleep(1)
            is_apsel_end = True
        return is_apsel_end

    def display_access_points(self, screen, ap_info):
        if False:
            for i in range(10):
                print('nop')
        '\n        Display information in the box window\n\n        :param self: A TuiApSel object\n        :type self: TuiApSel\n        :param screen: A curses window object\n        :type screen: _curses.curses.window\n        :param ap_info: An ApDisplayInfo object\n        :type ap_info: ApDisplayInfo\n        :return: None\n        :rtype: None\n        .. note: The display system is setup like the following:\n\n                 ----------------------------------------\n                 - (1,3)Options                         -\n                 -   (3,5)Header                        -\n                 - (4,3)****************************    -\n                 -      *       ^                  *    -\n                 -      *       |                  *    -\n                 -      *       |                  *    -\n                 -    < *       |----              *    -\n                 -    v *       |   v              *    -\n                 -    v *       |   v              *    -\n                 -    v *       |   v              *    -\n                 -    v *       v   v              *    -\n                 -    v ************v***************    -\n                 -    v             v      v            -\n                 -----v-------------v------v-------------\n                      v             v      v\n                      v             v      > max_window_length-5\n                      v             v\n                max_window_height-9 v\n                                    V\n                                    v--> box_height-2\n\n        '
        page_boundary = list(range(1 + ap_info.max_row * (ap_info.page_number - 1), ap_info.max_row + 1 + ap_info.max_row * (ap_info.page_number - 1)))
        ap_info.box.erase()
        ap_info.box.border(0)
        header_fmt = '{0:30} {1:16} {2:3} {3:4} {4:9} {5:5} {6:20}'
        header = header_fmt.format('ESSID', 'BSSID', 'CH', 'PWR', 'ENCR', 'CLIENTS', 'VENDOR')
        opt_str = 'Options:  [Esc] Quit  [Up Arrow] Move Up  [Down Arrow] Move Down'
        try:
            window_l = screen.getmaxyx()[1]
            screen.addstr(1, 3, display_string(window_l - 3, opt_str))
            screen.addstr(3, 5, display_string(window_l - 5, header))
        except curses.error:
            return
        for item_position in page_boundary:
            if self.total_ap_number == 0:
                display_str = 'No access point has been discovered yet!'
                try:
                    ap_info.box.addstr(1, 1, display_string(ap_info.max_l - 1, display_str), self.highlight_text)
                except curses.error:
                    return
            else:
                access_point = self.access_points[item_position - 1]
                vendor = self.mac_matcher.get_vendor_name(access_point.mac_address)
                display_text = '{0:30} {1:17} {2:2} {3:3}% {4:^8} {5:^5} {6:20}'.format(access_point.name, access_point.mac_address, access_point.channel, access_point.signal_strength, access_point.encryption, access_point.client_count, vendor)
                print_row_number = item_position - ap_info.max_row * (ap_info.page_number - 1)
                try:
                    if item_position == ap_info.pos:
                        ap_info.box.addstr(print_row_number, 2, display_string(ap_info.max_l - 2, display_text), self.highlight_text)
                    else:
                        ap_info.box.addstr(print_row_number, 2, display_string(ap_info.max_l - 2, display_text), self.normal_text)
                except curses.error:
                    return
                if item_position == self.total_ap_number:
                    break
        screen.refresh()
        ap_info.box.refresh()

class TuiMain(object):
    """
    TuiMain class to represent the main terminal window
    """

    def __init__(self):
        if False:
            while True:
                i = 10
        '\n        Construct the class\n        :param self: A TuiMain object\n        :type self: TuiMain\n        :return: None\n        :rtype: None\n        '
        self.blue_text = None
        self.orange_text = None
        self.yellow_text = None

    def gather_info(self, screen, info):
        if False:
            return 10
        '\n        Get the information from pywifiphisher and print them out\n        :param self: A TuiMain object\n        :param screen: A curses window object\n        :param info: A namedtuple of printing information\n        :type self: TuiMain\n        :type screen: _curses.curses.window\n        :type info: namedtuple\n        :return: None\n        :rtype: None\n        '
        try:
            curses.curs_set(0)
        except curses.error:
            pass
        screen.nodelay(True)
        curses.init_pair(1, curses.COLOR_BLUE, screen.getbkgd())
        curses.init_pair(2, curses.COLOR_YELLOW, screen.getbkgd())
        curses.init_pair(3, curses.COLOR_RED, screen.getbkgd())
        self.blue_text = curses.color_pair(1) | curses.A_BOLD
        self.yellow_text = curses.color_pair(2) | curses.A_BOLD
        self.red_text = curses.color_pair(3) | curses.A_BOLD
        while True:
            is_done = self.display_info(screen, info)
            if is_done:
                return

    def print_http_requests(self, screen, start_row_num, http_output):
        if False:
            while True:
                i = 10
        '\n        Print the http request on the main terminal\n        :param self: A TuiMain object\n        :type self: TuiMain\n        :param start_row_num: start line to print the http request\n        type start_row_num: int\n        :param http_output: string of the http requests\n        :type http_output: str\n        '
        requests = http_output.splitlines()
        match_str = '(.*\\s)(request from\\s)(.*)(\\sfor|with\\s)(.*)'
        for request in requests:
            match = re.match(match_str, request.decode('utf-8'))
            if match is None:
                continue
            request_type = match.group(1)
            request_from = match.group(2)
            ip_address = match.group(3)
            for_or_with = match.group(4)
            resource = match.group(5)
            start_col = 0
            screen.addstr(start_row_num, start_col, '[')
            start_col += 1
            screen.addstr(start_row_num, start_col, '*', self.yellow_text)
            start_col += 1
            screen.addstr(start_row_num, start_col, '] ')
            start_col += 2
            screen.addstr(start_row_num, start_col, request_type, self.yellow_text)
            start_col += len(request_type)
            screen.addstr(start_row_num, start_col, request_from)
            start_col += len(request_from)
            screen.addstr(start_row_num, start_col, ip_address, self.yellow_text)
            start_col += len(ip_address)
            screen.addstr(start_row_num, start_col, for_or_with)
            start_col += len(for_or_with)
            screen.addstr(start_row_num, start_col, resource, self.yellow_text)
            start_row_num += 1

    def display_info(self, screen, info):
        if False:
            while True:
                i = 10
        '\n        Print the information of Victims on the terminal\n        :param self: A TuiMain object\n        :param screen: A curses window object\n        :param info: A nameduple of printing information\n        :type self: TuiMain\n        :type screen: _curses.curses.window\n        :type info: namedtuple\n        :return True if users have pressed the Esc key\n        :rtype: bool\n        '
        accesspoint_instance = accesspoint.AccessPoint.get_instance()
        accesspoint_instance.read_connected_victims_file()
        is_done = False
        screen.erase()
        (_, max_window_length) = screen.getmaxyx()
        try:
            screen.addstr(0, max_window_length - 30, '|')
            screen.addstr(1, max_window_length - 30, '|')
            screen.addstr(1, max_window_length - 29, ' Wifiphisher ' + info.version, self.blue_text)
            screen.addstr(2, max_window_length - 30, '|' + ' ESSID: ' + info.essid)
            screen.addstr(3, max_window_length - 30, '|' + ' Channel: ' + info.channel)
            screen.addstr(4, max_window_length - 30, '|' + ' AP interface: ' + info.ap_iface)
            screen.addstr(5, max_window_length - 30, '|' + ' Options: [Esc] Quit')
            screen.addstr(6, max_window_length - 30, '|' + '_' * 29)
            screen.addstr(1, 0, 'Extensions feed: ', self.blue_text)
        except curses.error:
            pass
        if info.em:
            raw_num = 2
            for client in info.em.get_output()[-5:]:
                screen.addstr(raw_num, 0, client)
                raw_num += 1
        try:
            screen.addstr(7, 0, 'Connected Victims: ', self.blue_text)
            victims_instance = victim.Victims.get_instance()
            vict_dic = victims_instance.get_print_representation()
            row_counter = 8
            for key in vict_dic:
                screen.addstr(row_counter, 0, key, self.red_text)
                screen.addstr(row_counter, 22, vict_dic[key])
                row_counter += 1
            screen.addstr(13, 0, 'HTTP requests: ', self.blue_text)
            if os.path.isfile('/tmp/wifiphisher-webserver.tmp'):
                http_output = check_output(['tail', '-5', '/tmp/wifiphisher-webserver.tmp'])
                self.print_http_requests(screen, 14, http_output)
        except curses.error:
            pass
        if screen.getch() == 27:
            is_done = True
        if info.phishinghttp.terminate and info.args.quitonsuccess:
            is_done = True
        screen.refresh()
        return is_done

def display_string(w_len, target_line):
    if False:
        i = 10
        return i + 15
    '\n    Display the line base on the max length of window length\n    :param w_len: length of window\n    :param target_line: the target display string\n    :type w_len: int\n    :type target_line: str\n    :return: The final displaying string\n    :rtype: str\n    '
    return target_line if w_len >= len(target_line) else target_line[:w_len]

def line_splitter(num_of_words, line):
    if False:
        while True:
            i = 10
    '\n    Split line to the shorter lines\n    :param num_of_words: split the line into the line with lenth equeal\n    to num_of_words\n    :type num_of_words: int\n    :param line: A sentence\n    :type line: str\n    :return: tuple of shorter lines\n    :rtype: tuple\n    '
    pieces = line.split()
    return (' '.join(pieces[i:i + num_of_words]) for i in range(0, len(pieces), num_of_words))