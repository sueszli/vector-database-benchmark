version = __version__ = '0.39.0.6  Unreleased\n , VSeparator added (spelling error), added default key for one_line_progress_meter, auto-add keys to tables & trees, Graph.draw_image now uses image_data property instead of calling set_image, added theme_add_new, changed Remi call menu_item.set_on_click_listener to menu_item.onclick.connect so it can run with latest Remi'
port = 'PySimpleGUIWeb'
import sys
import datetime
import textwrap
import pickle
import threading
from queue import Queue
import remi
import logging
import traceback
import os
import base64, binascii
import mimetypes
from random import randint
import time
import pkg_resources
try:
    from io import StringIO
except:
    from cStringIO import StringIO
'\n    Welcome to the "core" PySimpleGUIWeb code....\n\n    This special port of the PySimpleGUI SDK to the browser is made possible by the magic of Remi \n\n    https://github.com/dddomodossola/remi \n\n    To be clear, PySimpleGUI would not be able to run in a web browser without this important GUI Framework\n    It may not be as widely known at tkinter or Qt, but it should be.  Just as those are the best of the desktop\n    GUI frameworks, Remi is THE framework for doing Web Page GUIs in Python.  Nothing else like it exists.                                       \n\n          :::::::::       ::::::::::         :::   :::       ::::::::::: \n         :+:    :+:      :+:               :+:+: :+:+:          :+:      \n        +:+    +:+      +:+              +:+ +:+:+ +:+         +:+       \n       +#++:++#:       +#++:++#         +#+  +:+  +#+         +#+        \n      +#+    +#+      +#+              +#+       +#+         +#+         \n     #+#    #+#      #+#              #+#       #+#         #+#          \n    ###    ###      ##########       ###       ###     ###########       \n\n'
g_time_start = 0
g_time_end = 0
g_time_delta = 0

def TimerStart():
    if False:
        return 10
    global g_time_start
    g_time_start = time.time()

def TimerStop():
    if False:
        return 10
    global g_time_delta, g_time_end
    g_time_end = time.time()
    g_time_delta = g_time_end - g_time_start
    print(g_time_delta * 1000)
DEFAULT_BASE64_ICON = b'iVBORw0KGgoAAAANSUhEUgAAACEAAAAgCAMAAACrZuH4AAAABGdBTUEAALGPC/xhBQAAAwBQTFRFAAAAMGmYMGqZMWqaMmubMmycM22dNGuZNm2bNm6bNG2dN26cNG6dNG6eNW+fN3CfOHCeOXGfNXCgNnGhN3KiOHOjOXSjOHSkOnWmOnamOnanPHSiPXakPnalO3eoPnimO3ioPHioPHmpPHmqPXqqPnurPnusPnytP3yuQHimQnurQn2sQH2uQX6uQH6vR32qRn+sSXujSHynTH2mTn+nSX6pQH6wTIGsTYKuTYSvQoCxQoCyRIK0R4S1RYS2Roa4SIe4SIe6SIi7Soq7SYm8SYq8Sou+TY2/UYStUYWvVIWtUYeyVoewUIi0VIizUI6+Vo+8WImxXJG5YI2xZI+xZ5CzZJC0ZpG1b5a3apW4aZm/cZi4dJ2/eJ69fJ+9XZfEZZnCZJzHaZ/Jdp/AeKTI/tM8/9Q7/9Q8/9Q9/9Q+/tQ//9VA/9ZA/9ZB/9ZC/9dD/9ZE/tdJ/9dK/9hF/9hG/9hH/9hI/9hJ/9hK/9lL/9pK/9pL/thO/9pM/9pN/9tO/9tP/9xP/tpR/9xQ/9xR/9xS/9xT/91U/91V/t1W/95W/95X/95Y/95Z/99a/99b/txf/txh/txk/t5l/t1q/t5v/+Bb/+Bc/+Bd/+Be/+Bf/+Bg/+Fh/+Fi/+Jh/+Ji/uJk/uJl/+Jm/+Rm/uJo/+Ro/+Rr/+Zr/+Vs/+Vu/+Zs/+Zu/uF0/uVw/+dw/+dz/+d2/uB5/uB6/uJ9/uR7/uR+/uV//+hx/+hy/+h0/+h2/+l4/+l7/+h8gKXDg6vLgazOhKzMiqrEj6/KhK/Qka/Hk7HJlLHJlLPMmLTLmbbOkLXSmLvXn77XoLrPpr/Tn8DaocLdpcHYrcjdssfZus/g/uOC/uOH/uaB/uWE/uaF/uWK/+qA/uqH/uqI/uuN/uyM/ueS/ueW/ueY/umQ/uqQ/uuS/uuW/uyU/uyX/uqa/uue/uye/uyf/u6f/uyq/u+r/u+t/vCm/vCp/vCu/vCy/vC2/vK2/vO8/vO/wtTjwtXlzdrl/vTA/vPQAAAAiNpY5gAAAQB0Uk5T////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////AFP3ByUAAAAJcEhZcwAAFw8AABcPASe7rwsAAAAYdEVYdFNvZnR3YXJlAHBhaW50Lm5ldCA0LjEuMWMqnEsAAAKUSURBVDhPhdB3WE1xHMdxt5JV0dANoUiyd8kqkey996xclUuTlEKidO3qVnTbhIyMW/bee5NskjJLmR/f3++cK/94vP76Ps/n/Zx7z6mE/6koJowcK154vvHOL/GsKCZXkUgkWlf4vWGWq5tsDz+JWIzSokAiqXGe7nWu3HxhEYof7fhOqp1GtptQuMruVhQdxZ05U5G47tYUHbQ4oah6Fg9Z4ubm7i57JhQjdHS0RSzUPoG17u6zZTKZh8c8XlytqW9YWUOH1LqFOZ6enl5ec+XybFb0rweM1tPTM6yuq6vLs0lYJJfLvb19fHwDWGF0jh5lYNAe4/QFemOwxtfXz8/fPyBgwVMqzAcCF4ybAZ2MRCexJGBhYGBQUHDw4u1UHDG1G2ZqB/Q1MTHmzAE+kpCwL1RghlTaBt/6SaXS2kx9YH1IaOjSZST8vfA9JtoDnSngGgL7wkg4WVkofA9mcF1Sx8zMzBK4v3wFiYiMVLxlEy9u21syFhYNmgN7IyJXEYViNZvEYoCVVWOmUVvgQVSUQqGIjolRFvOAFd8HWVs34VoA+6OjY2JjY5Vxm4BC1UuhGG5jY9OUaQXci1MqlfHx8YmqjyhOViW9ZsUN29akJRmPFwkJCZsTSXIpilJffXiTzorLXYgtcxRJKpUqKTklJQ0oSt9FP/EonxVdNY4jla1kK4q2ZB6mIr+AipvduzFUzMSOtLT09IyMzMxtJKug/F0u/6dTexAWDcXXLGEjapKjfsILOLKEuYiSnTQeYCt3UHhbwEHjGMrETfBJU5zq5dSTcXC8hLJccSWP2cgLXHPu7cQNAcpyxF1dyjehAKb0cSYUAOXCUw6V8OFPgevTXFymC+fPPLU677Nw/1X8A/AbfAKGulaqFlIAAAAASUVORK5CYII='
DEFAULT_WINDOW_ICON = 'default_icon.ico'
DEFAULT_ELEMENT_SIZE = (250, 26)
DEFAULT_BUTTON_ELEMENT_SIZE = (10, 1)
DEFAULT_MARGINS = (10, 5)
DEFAULT_ELEMENT_PADDING = (5, 3)
DEFAULT_AUTOSIZE_TEXT = True
DEFAULT_AUTOSIZE_BUTTONS = True
DEFAULT_FONT = ('Helvetica', 15)
DEFAULT_TEXT_JUSTIFICATION = 'left'
DEFAULT_BORDER_WIDTH = 1
DEFAULT_AUTOCLOSE_TIME = 3
DEFAULT_DEBUG_WINDOW_SIZE = (80, 20)
DEFAULT_OUTPUT_ELEMENT_SIZE = (40, 10)
DEFAULT_WINDOW_LOCATION = (None, None)
MAX_SCROLLED_TEXT_BOX_HEIGHT = 50
DEFAULT_TOOLTIP_TIME = 400
DEFAULT_PIXELS_TO_CHARS_SCALING = (10, 26)
DEFAULT_PIXEL_TO_CHARS_CUTOFF = 20
BLUES = ('#082567', '#0A37A3', '#00345B')
PURPLES = ('#480656', '#4F2398', '#380474')
GREENS = ('#01826B', '#40A860', '#96D2AB', '#00A949', '#003532')
YELLOWS = ('#F3FB62', '#F0F595')
TANS = ('#FFF9D5', '#F4EFCF', '#DDD8BA')
NICE_BUTTON_COLORS = ((GREENS[3], TANS[0]), ('#000000', '#FFFFFF'), ('#FFFFFF', '#000000'), (YELLOWS[0], PURPLES[1]), (YELLOWS[0], GREENS[3]), (YELLOWS[0], BLUES[2]))
COLOR_SYSTEM_DEFAULT = '1234567890'
DEFAULT_BUTTON_COLOR = ('white', BLUES[0])
OFFICIAL_PYSIMPLEGUI_BUTTON_COLOR = ('white', BLUES[0])
CURRENT_LOOK_AND_FEEL = 'DarkBlue3'
DEFAULT_ERROR_BUTTON_COLOR = ('#FFFFFF', '#FF0000')
DEFAULT_BACKGROUND_COLOR = None
DEFAULT_ELEMENT_BACKGROUND_COLOR = None
DEFAULT_ELEMENT_TEXT_COLOR = COLOR_SYSTEM_DEFAULT
DEFAULT_TEXT_ELEMENT_BACKGROUND_COLOR = None
DEFAULT_TEXT_COLOR = COLOR_SYSTEM_DEFAULT
DEFAULT_INPUT_ELEMENTS_COLOR = COLOR_SYSTEM_DEFAULT
DEFAULT_INPUT_TEXT_COLOR = COLOR_SYSTEM_DEFAULT
DEFAULT_SCROLLBAR_COLOR = None
TRANSPARENT_BUTTON = 'This constant has been depricated. You must set your button background = background it is on for it to be transparent appearing'
RELIEF_RAISED = 'raised'
RELIEF_SUNKEN = 'sunken'
RELIEF_FLAT = 'flat'
RELIEF_RIDGE = 'ridge'
RELIEF_GROOVE = 'groove'
RELIEF_SOLID = 'solid'
DEFAULT_PROGRESS_BAR_COLOR = (GREENS[0], '#D0D0D0')
DEFAULT_PROGRESS_BAR_COLOR_OFFICIAL = (GREENS[0], '#D0D0D0')
DEFAULT_PROGRESS_BAR_SIZE = (25, 20)
DEFAULT_PROGRESS_BAR_BORDER_WIDTH = 1
DEFAULT_PROGRESS_BAR_RELIEF = RELIEF_GROOVE
PROGRESS_BAR_STYLES = ('default', 'winnative', 'clam', 'alt', 'classic', 'vista', 'xpnative')
DEFAULT_PROGRESS_BAR_STYLE = 'default'
DEFAULT_METER_ORIENTATION = 'horizontal'
DEFAULT_SLIDER_ORIENTATION = 'vertical'
DEFAULT_SLIDER_BORDER_WIDTH = 1
DEFAULT_SLIDER_RELIEF = 0
DEFAULT_FRAME_RELIEF = 0
DEFAULT_LISTBOX_SELECT_MODE = 'extended'
SELECT_MODE_MULTIPLE = 'multiple'
LISTBOX_SELECT_MODE_MULTIPLE = 'multiple'
SELECT_MODE_BROWSE = 'browse'
LISTBOX_SELECT_MODE_BROWSE = 'browse'
SELECT_MODE_EXTENDED = 'extended'
LISTBOX_SELECT_MODE_EXTENDED = 'extended'
SELECT_MODE_SINGLE = 'single'
LISTBOX_SELECT_MODE_SINGLE = 'single'
SELECT_MODE_CONTIGUOUS = 'contiguous'
LISTBOX_SELECT_MODE_CONTIGUOUS = 'contiguous'
TABLE_SELECT_MODE_NONE = 0
TABLE_SELECT_MODE_BROWSE = 0
TABLE_SELECT_MODE_EXTENDED = 0
DEFAULT_TABLE_SECECT_MODE = TABLE_SELECT_MODE_EXTENDED
TITLE_LOCATION_TOP = 0
TITLE_LOCATION_BOTTOM = 0
TITLE_LOCATION_LEFT = 0
TITLE_LOCATION_RIGHT = 0
TITLE_LOCATION_TOP_LEFT = 0
TITLE_LOCATION_TOP_RIGHT = 0
TITLE_LOCATION_BOTTOM_LEFT = 0
TITLE_LOCATION_BOTTOM_RIGHT = 0
THEME_DEFAULT = 'default'
THEME_WINNATIVE = 'winnative'
THEME_CLAM = 'clam'
THEME_ALT = 'alt'
THEME_CLASSIC = 'classic'
THEME_VISTA = 'vista'
THEME_XPNATIVE = 'xpnative'
ThisRow = 555666777
MESSAGE_BOX_LINE_WIDTH = 60
EVENT_TIMEOUT = TIMEOUT_EVENT = TIMEOUT_KEY = '__TIMEOUT__'
WIN_CLOSED = WINDOW_CLOSED = None
WRITE_ONLY_KEY = '__WRITE ONLY__'
MENU_DISABLED_CHARACTER = '!'
MENU_KEY_SEPARATOR = '::'

class MyWindows:

    def __init__(self):
        if False:
            print('Hello World!')
        self._NumOpenWindows = 0
        self.user_defined_icon = None
        self.hidden_master_root = None

    def Decrement(self):
        if False:
            return 10
        self._NumOpenWindows -= 1 * (self._NumOpenWindows != 0)

    def Increment(self):
        if False:
            for i in range(10):
                print('nop')
        self._NumOpenWindows += 1
_my_windows = MyWindows()

def RGB(red, green, blue):
    if False:
        print('Hello World!')
    return '#%02x%02x%02x' % (red, green, blue)
BUTTON_TYPE_BROWSE_FOLDER = 1
BUTTON_TYPE_BROWSE_FILE = 2
BUTTON_TYPE_BROWSE_FILES = 21
BUTTON_TYPE_SAVEAS_FILE = 3
BUTTON_TYPE_CLOSES_WIN = 5
BUTTON_TYPE_CLOSES_WIN_ONLY = 6
BUTTON_TYPE_READ_FORM = 7
BUTTON_TYPE_REALTIME = 9
BUTTON_TYPE_CALENDAR_CHOOSER = 30
BUTTON_TYPE_COLOR_CHOOSER = 40
BROWSE_FILES_DELIMITER = ';'
ELEM_TYPE_TEXT = 'text'
ELEM_TYPE_INPUT_TEXT = 'input'
ELEM_TYPE_INPUT_COMBO = 'combo'
ELEM_TYPE_INPUT_OPTION_MENU = 'option menu'
ELEM_TYPE_INPUT_RADIO = 'radio'
ELEM_TYPE_INPUT_MULTILINE = 'multiline'
ELEM_TYPE_MULTILINE_OUTPUT = 'multioutput'
ELEM_TYPE_INPUT_CHECKBOX = 'checkbox'
ELEM_TYPE_INPUT_SPIN = 'spin'
ELEM_TYPE_BUTTON = 'button'
ELEM_TYPE_BUTTONMENU = 'buttonmenu'
ELEM_TYPE_IMAGE = 'image'
ELEM_TYPE_CANVAS = 'canvas'
ELEM_TYPE_FRAME = 'frame'
ELEM_TYPE_GRAPH = 'graph'
ELEM_TYPE_TAB = 'tab'
ELEM_TYPE_TAB_GROUP = 'tabgroup'
ELEM_TYPE_INPUT_SLIDER = 'slider'
ELEM_TYPE_INPUT_LISTBOX = 'listbox'
ELEM_TYPE_OUTPUT = 'output'
ELEM_TYPE_COLUMN = 'column'
ELEM_TYPE_MENUBAR = 'menubar'
ELEM_TYPE_PROGRESS_BAR = 'progressbar'
ELEM_TYPE_BLANK = 'blank'
ELEM_TYPE_TABLE = 'table'
ELEM_TYPE_TREE = 'tree'
ELEM_TYPE_ERROR = 'error'
ELEM_TYPE_SEPARATOR = 'separator'
POPUP_BUTTONS_YES_NO = 1
POPUP_BUTTONS_CANCELLED = 2
POPUP_BUTTONS_ERROR = 3
POPUP_BUTTONS_OK_CANCEL = 4
POPUP_BUTTONS_OK = 0
POPUP_BUTTONS_NO_BUTTONS = 5

class Element:

    def __init__(self, elem_type, size=(None, None), auto_size_text=None, font=None, background_color=None, text_color=None, key=None, pad=None, tooltip=None, visible=True, size_px=(None, None), metadata=None):
        if False:
            i = 10
            return i + 15
        if elem_type != ELEM_TYPE_GRAPH:
            self.Size = convert_tkinter_size_to_Wx(size)
        else:
            self.Size = size
        if size_px != (None, None):
            self.Size = size_px
        self.Type = elem_type
        self.AutoSizeText = auto_size_text
        self.Pad = pad
        if font is not None and type(font) is not str:
            self.Font = font
        elif font is not None:
            self.Font = font.split(' ')
        else:
            self.Font = font
        self.TKStringVar = None
        self.TKIntVar = None
        self.TKText = None
        self.TKEntry = None
        self.TKImage = None
        self.ParentForm = None
        self.ParentContainer = None
        self.TextInputDefault = None
        self.Position = (0, 0)
        self.BackgroundColor = background_color if background_color is not None else DEFAULT_ELEMENT_BACKGROUND_COLOR
        self.TextColor = text_color if text_color is not None else DEFAULT_ELEMENT_TEXT_COLOR
        self.Key = key
        self.Tooltip = tooltip
        self.TooltipObject = None
        self.Visible = visible
        self.metadata = metadata

    def _ChangedCallback(self, widget, *args):
        if False:
            print('Hello World!')
        self.ParentForm.LastButtonClicked = self.Key if self.Key is not None else ''
        self.ParentForm.MessageQueue.put(self.ParentForm.LastButtonClicked)

    def Update(self, widget, background_color=None, text_color=None, font=None, visible=None, disabled=None, tooltip=None):
        if False:
            while True:
                i = 10
        if font is not None:
            font_info = font_parse_string(font)
            widget.style['font-family'] = font_info[0]
            widget.style['font-size'] = '{}px'.format(font_info[1])
        if background_color not in (None, COLOR_SYSTEM_DEFAULT):
            widget.style['background-color'] = background_color
        if text_color not in (None, COLOR_SYSTEM_DEFAULT):
            widget.style['color'] = text_color
        if disabled:
            widget.set_enabled(False)
        elif disabled is False:
            widget.set_enabled(True)
        if visible is False:
            widget.attributes['hidden'] = 'true'
        elif visible is True:
            del widget.attributes['hidden']
        if tooltip is not None:
            widget.attributes['title'] = tooltip
        if visible is False:
            widget.attributes['hidden'] = 'true'
        elif visible is True:
            del widget.attributes['hidden']

    def __call__(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        '\n        Makes it possible to "call" an already existing element.  When you do make the "call", it actually calls\n        the Update method for the element.\n        Example:    If this text element was in your layout:\n                    sg.Text(\'foo\', key=\'T\')\n                    Then you can call the Update method for that element by writing:\n                    window.FindElement(\'T\')(\'new text value\')\n\n        :param args:\n        :param kwargs:\n        :return:\n        '
        return self.Update(*args, **kwargs)

class InputText(Element):

    def __init__(self, default_text='', size=(None, None), disabled=False, password_char='', justification=None, background_color=None, text_color=None, font=None, tooltip=None, change_submits=False, enable_events=False, do_not_clear=True, key=None, focus=False, pad=None, visible=True, size_px=(None, None)):
        if False:
            return 10
        '\n        Input a line of text Element\n        :param default_text: Default value to display\n        :param size: Size of field in characters\n        :param password_char: If non-blank, will display this character for every character typed\n        :param background_color: Color for Element. Text or RGB Hex\n        '
        self.DefaultText = default_text
        self.PasswordCharacter = password_char
        bg = background_color if background_color is not None else DEFAULT_INPUT_ELEMENTS_COLOR
        fg = text_color if text_color is not None else DEFAULT_INPUT_TEXT_COLOR
        self.Focus = focus
        self.do_not_clear = do_not_clear
        self.Justification = justification or 'left'
        self.Disabled = disabled
        self.ChangeSubmits = change_submits or enable_events
        self.QT_QLineEdit = None
        self.ValueWasChanged = False
        self.Widget = None
        super().__init__(ELEM_TYPE_INPUT_TEXT, size=size, background_color=bg, text_color=fg, key=key, pad=pad, font=font, tooltip=tooltip, visible=visible, size_px=size_px)

    def _InputTextCallback(self, widget, key, keycode, ctrl, shift, alt):
        if False:
            for i in range(10):
                print('nop')
        self.ParentForm.LastButtonClicked = key
        self.ParentForm.MessageQueue.put(self.ParentForm.LastButtonClicked)
        widget.set_value(widget.get_value() + key)
        return (key, keycode, ctrl, shift, alt)

    def Update(self, value=None, disabled=None, select=None, background_color=None, text_color=None, font=None, visible=None):
        if False:
            i = 10
            return i + 15
        if value is not None:
            self.Widget.set_value(str(value))
        if disabled is True:
            self.Widget.set_enabled(False)
        elif disabled is False:
            self.Widget.set_enabled(True)

    def Get(self):
        if False:
            while True:
                i = 10
        return self.Widget.get_value()
    get = Get
    update = Update

    class TextInput_raw_onkeyup(remi.gui.TextInput):

        @remi.gui.decorate_set_on_listener('(self, emitter, key, keycode, ctrl, shift, alt)')
        @remi.gui.decorate_event_js("var params={};params['key']=event.key;\n                params['keycode']=(event.which||event.keyCode);\n                params['ctrl']=event.ctrlKey;\n                params['shift']=event.shiftKey;\n                params['alt']=event.altKey;\n                sendCallbackParam('%(emitter_identifier)s','%(event_name)s',params);\n                event.stopPropagation();event.preventDefault();return false;")
        def onkeyup(self, key, keycode, ctrl, shift, alt):
            if False:
                return 10
            return (key, keycode, ctrl, shift, alt)

        @remi.gui.decorate_set_on_listener('(self, emitter, key, keycode, ctrl, shift, alt)')
        @remi.gui.decorate_event_js("var params={};params['key']=event.key;\n                params['keycode']=(event.which||event.keyCode);\n                params['ctrl']=event.ctrlKey;\n                params['shift']=event.shiftKey;\n                params['alt']=event.altKey;\n                sendCallbackParam('%(emitter_identifier)s','%(event_name)s',params);\n                event.stopPropagation();event.preventDefault();return false;")
        def onkeydown(self, key, keycode, ctrl, shift, alt):
            if False:
                for i in range(10):
                    print('nop')
            return (key, keycode, ctrl, shift, alt)
In = InputText
Input = InputText
I = InputText

class Combo(Element):

    def __init__(self, values, default_value=None, size=(None, None), auto_size_text=None, background_color=None, text_color=None, change_submits=False, enable_events=False, disabled=False, key=None, pad=None, tooltip=None, readonly=False, visible_items=10, font=None, auto_complete=True, visible=True, size_px=(None, None)):
        if False:
            return 10
        '\n        Input Combo Box Element (also called Dropdown box)\n        :param values:\n        :param size: Size of field in characters\n        :param auto_size_text: True if should shrink field to fit the default text\n        :param background_color: Color for Element. Text or RGB Hex\n        '
        self.Values = [str(v) for v in values]
        self.DefaultValue = default_value
        self.ChangeSubmits = change_submits or enable_events
        self.Disabled = disabled
        self.Readonly = readonly
        bg = background_color if background_color else DEFAULT_INPUT_ELEMENTS_COLOR
        fg = text_color if text_color is not None else DEFAULT_INPUT_TEXT_COLOR
        self.VisibleItems = visible_items
        self.AutoComplete = auto_complete
        self.Widget = None
        super().__init__(ELEM_TYPE_INPUT_COMBO, size=size, auto_size_text=auto_size_text, background_color=bg, text_color=fg, key=key, pad=pad, tooltip=tooltip, font=font or DEFAULT_FONT, visible=visible, size_px=size_px)

    def Update(self, value=None, values=None, set_to_index=None, disabled=None, readonly=None, background_color=None, text_color=None, font=None, visible=None):
        if False:
            while True:
                i = 10
        if values is not None:
            self.Widget.empty()
            for (i, item) in enumerate(values):
                self.Widget.append(value=item, key=str(i))
        if value:
            self.Widget.select_by_value(value)
        if set_to_index is not None:
            try:
                self.Widget.select_by_key(str(set_to_index))
            except:
                pass
        super().Update(self.Widget, background_color=background_color, text_color=text_color, font=font, visible=visible, disabled=disabled)
    update = Update
InputCombo = Combo
DropDown = Combo
Drop = Combo

class OptionMenu(Element):

    def __init__(self, values, default_value=None, size=(None, None), disabled=False, auto_size_text=None, background_color=None, text_color=None, key=None, pad=None, tooltip=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        InputOptionMenu\n        :param values:\n        :param default_value:\n        :param size:\n        :param disabled:\n        :param auto_size_text:\n        :param background_color:\n        :param text_color:\n        :param key:\n        :param pad:\n        :param tooltip:\n        '
        self.Values = values
        self.DefaultValue = default_value
        self.TKOptionMenu = None
        self.Disabled = disabled
        bg = background_color if background_color else DEFAULT_INPUT_ELEMENTS_COLOR
        fg = text_color if text_color is not None else DEFAULT_INPUT_TEXT_COLOR
        super().__init__(ELEM_TYPE_INPUT_OPTION_MENU, size=size, auto_size_text=auto_size_text, background_color=bg, text_color=fg, key=key, pad=pad, tooltip=tooltip)

    def Update(self, value=None, values=None, disabled=None):
        if False:
            for i in range(10):
                print('nop')
        if values is not None:
            self.Values = values
        if self.Values is not None:
            for (index, v) in enumerate(self.Values):
                if v == value:
                    try:
                        self.TKStringVar.set(value)
                    except:
                        pass
                    self.DefaultValue = value
                    break
        if disabled == True:
            self.TKOptionMenu['state'] = 'disabled'
        elif disabled == False:
            self.TKOptionMenu['state'] = 'normal'
InputOptionMenu = OptionMenu

class Listbox(Element):

    def __init__(self, values, default_values=None, select_mode=None, change_submits=False, enable_events=False, bind_return_key=False, size=(None, None), disabled=False, auto_size_text=None, font=None, background_color=None, text_color=None, key=None, pad=None, tooltip=None, visible=True, size_px=(None, None)):
        if False:
            for i in range(10):
                print('nop')
        '\n\n        :param values:\n        :param default_values:\n        :param select_mode:\n        :param change_submits:\n        :param enable_events:\n        :param bind_return_key:\n        :param size:\n        :param disabled:\n        :param auto_size_text:\n        :param font:\n        :param background_color:\n        :param text_color:\n        :param key:\n        :param pad:\n        :param tooltip:\n        :param visible:\n        :param size_px:\n        '
        self.Values = values
        self.DefaultValues = default_values
        self.TKListbox = None
        self.ChangeSubmits = change_submits or enable_events
        self.BindReturnKey = bind_return_key
        self.Disabled = disabled
        if select_mode == LISTBOX_SELECT_MODE_BROWSE:
            self.SelectMode = SELECT_MODE_BROWSE
        elif select_mode == LISTBOX_SELECT_MODE_EXTENDED:
            self.SelectMode = SELECT_MODE_EXTENDED
        elif select_mode == LISTBOX_SELECT_MODE_MULTIPLE:
            self.SelectMode = SELECT_MODE_MULTIPLE
        elif select_mode == LISTBOX_SELECT_MODE_SINGLE:
            self.SelectMode = SELECT_MODE_SINGLE
        elif select_mode == LISTBOX_SELECT_MODE_CONTIGUOUS:
            self.SelectMode = SELECT_MODE_CONTIGUOUS
        else:
            self.SelectMode = DEFAULT_LISTBOX_SELECT_MODE
        bg = background_color if background_color else DEFAULT_INPUT_ELEMENTS_COLOR
        fg = text_color if text_color is not None else DEFAULT_INPUT_TEXT_COLOR
        self.Widget = None
        tsize = size
        if size[0] is not None and size[0] < 100:
            tsize = (size[0] * DEFAULT_PIXELS_TO_CHARS_SCALING[0], size[1] * DEFAULT_PIXELS_TO_CHARS_SCALING[1])
        super().__init__(ELEM_TYPE_INPUT_LISTBOX, size=tsize, auto_size_text=auto_size_text, font=font, background_color=bg, text_color=fg, key=key, pad=pad, tooltip=tooltip, visible=visible, size_px=size_px)

    def Update(self, values=None, disabled=None, set_to_index=None, background_color=None, text_color=None, font=None, visible=None):
        if False:
            print('Hello World!')
        if values is not None:
            self.Values = values
            self.Widget.empty()
            for item in values:
                self.Widget.append(remi.gui.ListItem(item))
        super().Update(self.Widget, background_color=background_color, text_color=text_color, font=font, visible=visible, disabled=disabled)
        return

    def GetListValues(self):
        if False:
            i = 10
            return i + 15
        return self.Values
    get_list_values = GetListValues
    update = Update

class Radio(Element):

    def __init__(self, text, group_id, default=False, disabled=False, size=(None, None), auto_size_text=None, background_color=None, text_color=None, font=None, key=None, pad=None, tooltip=None, change_submits=False):
        if False:
            return 10
        '\n        Radio Button Element\n        :param text:\n        :param group_id:\n        :param default:\n        :param disabled:\n        :param size:\n        :param auto_size_text:\n        :param background_color:\n        :param text_color:\n        :param font:\n        :param key:\n        :param pad:\n        :param tooltip:\n        :param change_submits:\n        '
        self.InitialState = default
        self.Text = text
        self.TKRadio = None
        self.GroupID = group_id
        self.Value = None
        self.Disabled = disabled
        self.TextColor = text_color or DEFAULT_TEXT_COLOR
        self.ChangeSubmits = change_submits
        print('*** WARNING - Radio Buttons are not yet available on PySimpleGUIWeb ***')
        super().__init__(ELEM_TYPE_INPUT_RADIO, size=size, auto_size_text=auto_size_text, font=font, background_color=background_color, text_color=self.TextColor, key=key, pad=pad, tooltip=tooltip)

    def Update(self, value=None, disabled=None):
        if False:
            i = 10
            return i + 15
        print('*** NOT IMPLEMENTED ***')
        location = EncodeRadioRowCol(self.Position[0], self.Position[1])
        if value is not None:
            try:
                self.TKIntVar.set(location)
            except:
                pass
            self.InitialState = value
        if disabled == True:
            self.TKRadio['state'] = 'disabled'
        elif disabled == False:
            self.TKRadio['state'] = 'normal'
    update = Update

class Checkbox(Element):

    def __init__(self, text, default=False, size=(None, None), auto_size_text=None, font=None, background_color=None, text_color=None, change_submits=False, enable_events=False, disabled=False, key=None, pad=None, tooltip=None, visible=True, size_px=(None, None)):
        if False:
            while True:
                i = 10
        '\n        Checkbox Element\n        :param text:\n        :param default:\n        :param size:\n        :param auto_size_text:\n        :param font:\n        :param background_color:\n        :param text_color:\n        :param change_submits:\n        :param disabled:\n        :param key:\n        :param pad:\n        :param tooltip:\n        '
        self.Text = text
        self.InitialState = default
        self.Disabled = disabled
        self.TextColor = text_color if text_color else DEFAULT_TEXT_COLOR
        self.ChangeSubmits = change_submits or enable_events
        self.Widget = None
        super().__init__(ELEM_TYPE_INPUT_CHECKBOX, size=size, auto_size_text=auto_size_text, font=font, background_color=background_color, text_color=self.TextColor, key=key, pad=pad, tooltip=tooltip, visible=visible, size_px=size_px)

    def _ChangedCallback(self, widget, value):
        if False:
            print('Hello World!')
        self.ParentForm.LastButtonClicked = self.Key
        self.ParentForm.MessageQueue.put(self.ParentForm.LastButtonClicked)

    def Get(self):
        if False:
            for i in range(10):
                print('nop')
        return self.Widget.get_value()

    def Update(self, value=None, disabled=None):
        if False:
            print('Hello World!')
        if value is not None:
            self.Widget.set_value(value)
        if disabled == True:
            self.Widget.set_enabled(False)
        elif disabled == False:
            self.Widget.set_enabled(True)
    get = Get
    update = Update
CB = Checkbox
CBox = Checkbox
Check = Checkbox

class Spin(Element):

    def __init__(self, values, initial_value=None, disabled=False, change_submits=False, enable_events=False, size=(None, None), readonly=True, auto_size_text=None, font=None, background_color=None, text_color=None, key=None, pad=None, tooltip=None, visible=True, size_px=(None, None)):
        if False:
            print('Hello World!')
        '\n        Spinner Element\n        :param values:\n        :param initial_value:\n        :param disabled:\n        :param change_submits:\n        :param size:\n        :param auto_size_text:\n        :param font:\n        :param background_color:\n        :param text_color:\n        :param key:\n        :param pad:\n        :param tooltip:\n        '
        self.Values = values
        self.DefaultValue = initial_value or values[0]
        self.ChangeSubmits = change_submits or enable_events
        self.Disabled = disabled
        bg = background_color if background_color else DEFAULT_INPUT_ELEMENTS_COLOR
        fg = text_color if text_color is not None else DEFAULT_INPUT_TEXT_COLOR
        self.CurrentValue = self.DefaultValue
        self.ReadOnly = readonly
        self.Widget = None
        super().__init__(ELEM_TYPE_INPUT_SPIN, size, auto_size_text, font=font, background_color=bg, text_color=fg, key=key, pad=pad, tooltip=tooltip, visible=visible, size_px=size_px)
        return

    def Update(self, value=None, values=None, disabled=None, background_color=None, text_color=None, font=None, visible=None):
        if False:
            i = 10
            return i + 15
        if value is not None:
            self.Widget.set_value(value)
        super().Update(self.Widget, background_color=background_color, text_color=text_color, font=font, visible=visible)

    def Get(self):
        if False:
            for i in range(10):
                print('nop')
        return self.Widget.get_value()
    get = Get
    update = Update

class Multiline(Element):

    def __init__(self, default_text='', enter_submits=False, disabled=False, autoscroll=False, size=(None, None), auto_size_text=None, background_color=None, text_color=None, change_submits=False, enable_events=False, do_not_clear=True, key=None, write_only=False, focus=False, font=None, pad=None, tooltip=None, visible=True, size_px=(None, None)):
        if False:
            while True:
                i = 10
        '\n        Multiline Element\n        :param default_text:\n        :param enter_submits:\n        :param disabled:\n        :param autoscroll:\n        :param size:\n        :param auto_size_text:\n        :param background_color:\n        :param text_color:\n        :param do_not_clear:\n        :param key:\n        :param focus:\n        :param pad:\n        :param tooltip:\n        :param font:\n        '
        self.DefaultText = default_text
        self.EnterSubmits = enter_submits
        bg = background_color if background_color else DEFAULT_INPUT_ELEMENTS_COLOR
        self.Focus = focus
        self.do_not_clear = do_not_clear
        fg = text_color if text_color is not None else DEFAULT_INPUT_TEXT_COLOR
        self.Autoscroll = autoscroll
        self.Disabled = disabled
        self.ChangeSubmits = change_submits or enable_events
        self.WriteOnly = write_only
        if size[0] is not None and size[0] < 100:
            size = (size[0] * DEFAULT_PIXELS_TO_CHARS_SCALING[0], size[1] * DEFAULT_PIXELS_TO_CHARS_SCALING[1])
        self.Widget = None
        super().__init__(ELEM_TYPE_INPUT_MULTILINE, size=size, auto_size_text=auto_size_text, background_color=bg, text_color=fg, key=key, pad=pad, tooltip=tooltip, font=font or DEFAULT_FONT, visible=visible, size_px=size_px)
        return

    def _InputTextCallback(self, widget: remi.Widget, value, keycode):
        if False:
            while True:
                i = 10
        self.ParentForm.LastButtonClicked = chr(int(keycode))
        self.ParentForm.MessageQueue.put(self.ParentForm.LastButtonClicked)

    def Update(self, value=None, disabled=None, append=False, background_color=None, text_color=None, font=None, visible=None, autoscroll=None):
        if False:
            return 10
        if value is not None and (not append):
            self.Widget.set_value(value)
        elif value is not None and append:
            text = self.Widget.get_value() + str(value)
            self.Widget.set_value(text)
        super().Update(self.Widget, background_color=background_color, text_color=text_color, font=font, visible=visible)

    def print(self, *args, end=None, sep=None, text_color=None, background_color=None):
        if False:
            return 10
        '\n        Print like Python normally prints except route the output to a multline element and also add colors if desired\n\n        :param args: List[Any] The arguments to print\n        :param end: (str) The end char to use just like print uses\n        :param sep: (str) The separation character like print uses\n        :param text_color: The color of the text\n        :param background_color: The background color of the line\n        '
        _print_to_element(self, *args, end=end, sep=sep, text_color=text_color, background_color=background_color)
    update = Update
ML = Multiline
MLine = Multiline

class MultilineOutput(Element):

    def __init__(self, default_text='', enter_submits=False, disabled=False, autoscroll=False, size=(None, None), auto_size_text=None, background_color=None, text_color=None, change_submits=False, enable_events=False, do_not_clear=True, key=None, focus=False, font=None, pad=None, tooltip=None, visible=True, size_px=(None, None)):
        if False:
            i = 10
            return i + 15
        '\n        Multiline Element\n        :param default_text:\n        :param enter_submits:\n        :param disabled:\n        :param autoscroll:\n        :param size:\n        :param auto_size_text:\n        :param background_color:\n        :param text_color:\n        :param do_not_clear:\n        :param key:\n        :param focus:\n        :param pad:\n        :param tooltip:\n        :param font:\n        '
        self.DefaultText = default_text
        self.EnterSubmits = enter_submits
        bg = background_color if background_color else DEFAULT_INPUT_ELEMENTS_COLOR
        self.Focus = focus
        self.do_not_clear = do_not_clear
        fg = text_color if text_color is not None else DEFAULT_INPUT_TEXT_COLOR
        self.Autoscroll = autoscroll
        self.Disabled = disabled
        self.ChangeSubmits = change_submits or enable_events
        tsize = size
        if size[0] is not None and size[0] < 100:
            tsize = (size[0] * DEFAULT_PIXELS_TO_CHARS_SCALING[0], size[1] * DEFAULT_PIXELS_TO_CHARS_SCALING[1])
        self.Widget = None
        self.CurrentValue = ''
        super().__init__(ELEM_TYPE_MULTILINE_OUTPUT, size=tsize, auto_size_text=auto_size_text, background_color=bg, text_color=fg, key=key, pad=pad, tooltip=tooltip, font=font or DEFAULT_FONT, visible=visible, size_px=size_px)
        return

    def Update(self, value=None, disabled=None, append=False, background_color=None, text_color=None, font=None, visible=None, autoscroll=None):
        if False:
            return 10
        autoscroll = self.Autoscroll if autoscroll is None else autoscroll
        if value is not None and (not append):
            self.Widget.set_value(str(value))
            self.CurrentValue = str(value)
        elif value is not None and append:
            self.CurrentValue = self.CurrentValue + str(value)
            self.Widget.set_value(self.CurrentValue)
        self.Widget._set_updated()
        app = self.ParentForm.App
        if hasattr(app, 'websockets'):
            app.execute_javascript('element=document.getElementById("%(id)s"); element.innerHTML=`%(content)s`; if(%(autoscroll)s){element.scrollTop=999999;} ' % {'id': self.Widget.identifier, 'content': self.Widget.get_value(), 'autoscroll': 'true' if autoscroll else 'false'})
        super().Update(self.Widget, background_color=background_color, text_color=text_color, font=font, visible=visible)

    def print(self, *args, end=None, sep=None, text_color=None, background_color=None):
        if False:
            i = 10
            return i + 15
        '\n        Print like Python normally prints except route the output to a multline element and also add colors if desired\n\n        :param args: List[Any] The arguments to print\n        :param end: (str) The end char to use just like print uses\n        :param sep: (str) The separation character like print uses\n        :param text_color: The color of the text\n        :param background_color: The background color of the line\n        '
        _print_to_element(self, *args, end=end, sep=sep, text_color=text_color, background_color=background_color)
    update = Update

class Text(Element):

    def __init__(self, text='', size=(None, None), auto_size_text=None, click_submits=None, enable_events=False, relief=None, border_width=None, font=None, text_color=None, background_color=None, justification=None, pad=None, margins=None, key=None, tooltip=None, visible=True, size_px=(None, None), metadata=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        Text\n        :param text:\n        :param size:\n        :param auto_size_text:\n        :param click_submits:\n        :param enable_events:\n        :param relief:\n        :param font:\n        :param text_color:\n        :param background_color:\n        :param justification:\n        :param pad:\n        :param margins:\n        :param key:\n        :param tooltip:\n        :param visible:\n        :param size_px:\n        '
        self.DisplayText = str(text)
        self.TextColor = text_color if text_color else DEFAULT_TEXT_COLOR
        self.Justification = justification
        self.Relief = relief
        self.ClickSubmits = click_submits or enable_events
        self.Margins = margins
        self.size_px = size_px
        if background_color is None:
            bg = DEFAULT_TEXT_ELEMENT_BACKGROUND_COLOR
        else:
            bg = background_color
        pixelsize = size
        if size[1] is not None and size[1] < 10:
            pixelsize = (size[0] * 10, size[1] * 20)
        self.BorderWidth = border_width if border_width is not None else DEFAULT_BORDER_WIDTH
        self.Disabled = False
        self.Widget = None
        super().__init__(ELEM_TYPE_TEXT, pixelsize, auto_size_text, background_color=bg, font=font if font else DEFAULT_FONT, text_color=self.TextColor, pad=pad, key=key, tooltip=tooltip, size_px=size_px, visible=visible, metadata=metadata)
        return

    def Update(self, value=None, background_color=None, text_color=None, font=None, visible=None):
        if False:
            i = 10
            return i + 15
        if value is not None:
            self.Widget.set_text(str(value))
        super().Update(self.Widget, background_color=background_color, text_color=text_color, font=font, visible=visible)
    update = Update
Txt = Text
T = Text

class Output(Element):

    def __init__(self, size=(None, None), background_color=None, text_color=None, pad=None, font=None, tooltip=None, key=None, visible=True, size_px=(None, None), disabled=False):
        if False:
            i = 10
            return i + 15
        '\n        Output Element\n        :param size:\n        :param background_color:\n        :param text_color:\n        :param pad:\n        :param font:\n        :param tooltip:\n        :param key:\n        '
        bg = background_color if background_color else DEFAULT_INPUT_ELEMENTS_COLOR
        fg = text_color if text_color is not None else 'black' if DEFAULT_INPUT_TEXT_COLOR == COLOR_SYSTEM_DEFAULT else DEFAULT_INPUT_TEXT_COLOR
        self.Disabled = disabled
        self.Widget = None
        if size_px == (None, None) and size == (None, None):
            size = DEFAULT_OUTPUT_ELEMENT_SIZE
        if size[0] is not None and size[0] < 100:
            size = (size[0] * DEFAULT_PIXELS_TO_CHARS_SCALING[0], size[1] * DEFAULT_PIXELS_TO_CHARS_SCALING[1])
        super().__init__(ELEM_TYPE_OUTPUT, size=size, size_px=size_px, visible=visible, background_color=bg, text_color=fg, pad=pad, font=font, tooltip=tooltip, key=key)

    def Update(self, value=None, disabled=None, append=False, background_color=None, text_color=None, font=None, visible=None):
        if False:
            i = 10
            return i + 15
        if value is not None and (not append):
            self.Widget.set_value(str(value))
            self.CurrentValue = str(value)
        elif value is not None and append:
            self.CurrentValue = self.CurrentValue + '\n' + str(value)
            self.Widget.set_value(self.CurrentValue)
        self.Widget._set_updated()
        app = self.ParentForm.App
        if hasattr(app, 'websockets'):
            app.execute_javascript('element=document.getElementById("%(id)s"); element.innerHTML=`%(content)s`; element.scrollTop=999999; ' % {'id': self.Widget.identifier, 'content': self.Widget.get_value()})
        super().Update(self.Widget, background_color=background_color, text_color=text_color, font=font, visible=visible)
    update = Update

class Button(Element):

    def __init__(self, button_text='', button_type=BUTTON_TYPE_READ_FORM, target=(None, None), tooltip=None, file_types=(('ALL Files', '*'),), initial_folder=None, disabled=False, change_submits=False, enable_events=False, image_filename=None, image_data=None, image_size=(None, None), image_subsample=None, border_width=None, size=(None, None), auto_size_button=None, button_color=None, font=None, bind_return_key=False, focus=False, pad=None, key=None, visible=True, size_px=(None, None)):
        if False:
            while True:
                i = 10
        '\n        Button Element\n        :param button_text:\n        :param button_type:\n        :param target:\n        :param tooltip:\n        :param file_types:\n        :param initial_folder:\n        :param disabled:\n        :param image_filename:\n        :param image_size:\n        :param image_subsample:\n        :param border_width:\n        :param size:\n        :param auto_size_button:\n        :param button_color:\n        :param default_value:\n        :param font:\n        :param bind_return_key:\n        :param focus:\n        :param pad:\n        :param key:\n        '
        self.AutoSizeButton = auto_size_button
        self.BType = button_type
        self.FileTypes = file_types
        self.TKButton = None
        self.Target = target
        self.ButtonText = str(button_text)
        self.ButtonColor = button_color if button_color else DEFAULT_BUTTON_COLOR
        self.TextColor = self.ButtonColor[0]
        self.BackgroundColor = self.ButtonColor[1]
        self.ImageFilename = image_filename
        self.ImageData = image_data
        self.ImageSize = image_size
        self.ImageSubsample = image_subsample
        self.UserData = None
        self.BorderWidth = border_width if border_width is not None else DEFAULT_BORDER_WIDTH
        self.BindReturnKey = bind_return_key
        self.Focus = focus
        self.TKCal = None
        self.CalendarCloseWhenChosen = None
        self.DefaultDate_M_D_Y = (None, None, None)
        self.InitialFolder = initial_folder
        self.Disabled = disabled
        self.ChangeSubmits = change_submits or enable_events
        self.QT_QPushButton = None
        self.ColorChosen = None
        self.Relief = None
        self.Widget = None
        super().__init__(ELEM_TYPE_BUTTON, size=size, font=font, pad=pad, key=key, tooltip=tooltip, text_color=self.TextColor, background_color=self.BackgroundColor, visible=visible, size_px=size_px)
        return

    def _ButtonCallBack(self, event):
        if False:
            print('Hello World!')
        target = self.Target
        target_element = None
        if target[0] == ThisRow:
            target = [self.Position[0], target[1]]
            if target[1] < 0:
                target[1] = self.Position[1] + target[1]
        strvar = None
        should_submit_window = False
        if target == (None, None):
            strvar = self.TKStringVar
        else:
            if not isinstance(target, str):
                if target[0] < 0:
                    target = [self.Position[0] + target[0], target[1]]
                target_element = self.ParentContainer._GetElementAtLocation(target)
            else:
                target_element = self.ParentForm.FindElement(target)
            try:
                strvar = target_element.TKStringVar
            except:
                pass
            try:
                if target_element.ChangeSubmits:
                    should_submit_window = True
            except:
                pass
        filetypes = (('ALL Files', '*'),) if self.FileTypes is None else self.FileTypes
        if self.BType == BUTTON_TYPE_BROWSE_FOLDER:
            wx_types = convert_tkinter_filetypes_to_wx(self.FileTypes)
            if self.InitialFolder:
                dialog = wx.DirDialog(self.ParentForm.MasterFrame, style=wx.FD_OPEN)
            else:
                dialog = wx.DirDialog(self.ParentForm.MasterFrame)
            folder_name = ''
            if dialog.ShowModal() == wx.ID_OK:
                folder_name = dialog.GetPath()
            if folder_name != '':
                if target_element.Type == ELEM_TYPE_BUTTON:
                    target_element.FileOrFolderName = folder_name
                else:
                    target_element.Update(folder_name)
        elif self.BType == BUTTON_TYPE_BROWSE_FILE:
            qt_types = convert_tkinter_filetypes_to_wx(self.FileTypes)
            if self.InitialFolder:
                dialog = wx.FileDialog(self.ParentForm.MasterFrame, defaultDir=self.InitialFolder, wildcard=qt_types, style=wx.FD_OPEN)
            else:
                dialog = wx.FileDialog(self.ParentForm.MasterFrame, wildcard=qt_types, style=wx.FD_OPEN)
            file_name = ''
            if dialog.ShowModal() == wx.ID_OK:
                file_name = dialog.GetPath()
            else:
                file_name = ''
            if file_name != '':
                if target_element.Type == ELEM_TYPE_BUTTON:
                    target_element.FileOrFolderName = file_name
                else:
                    target_element.Update(file_name)
        elif self.BType == BUTTON_TYPE_BROWSE_FILES:
            qt_types = convert_tkinter_filetypes_to_wx(self.FileTypes)
            if self.InitialFolder:
                dialog = wx.FileDialog(self.ParentForm.MasterFrame, defaultDir=self.InitialFolder, wildcard=qt_types, style=wx.FD_MULTIPLE)
            else:
                dialog = wx.FileDialog(self.ParentForm.MasterFrame, wildcard=qt_types, style=wx.FD_MULTIPLE)
            file_names = ''
            if dialog.ShowModal() == wx.ID_OK:
                file_names = dialog.GetPaths()
            else:
                file_names = ''
            if file_names != '':
                file_names = BROWSE_FILES_DELIMITER.join(file_names)
                if target_element.Type == ELEM_TYPE_BUTTON:
                    target_element.FileOrFolderName = file_names
                else:
                    target_element.Update(file_names)
        elif self.BType == BUTTON_TYPE_SAVEAS_FILE:
            qt_types = convert_tkinter_filetypes_to_wx(self.FileTypes)
            if self.InitialFolder:
                dialog = wx.FileDialog(self.ParentForm.MasterFrame, defaultDir=self.InitialFolder, wildcard=qt_types, style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT)
            else:
                dialog = wx.FileDialog(self.ParentForm.MasterFrame, wildcard=qt_types, style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT)
            file_name = ''
            if dialog.ShowModal() == wx.ID_OK:
                file_name = dialog.GetPath()
            else:
                file_name = ''
            if file_name != '':
                if target_element.Type == ELEM_TYPE_BUTTON:
                    target_element.FileOrFolderName = file_name
                else:
                    target_element.Update(file_name)
        elif self.BType == BUTTON_TYPE_COLOR_CHOOSER:
            qcolor = QColorDialog.getColor()
            rgb_color = qcolor.getRgb()
            color = '#' + ''.join(('%02x' % i for i in rgb_color[:3]))
            if self.Target == (None, None):
                self.FileOrFolderName = color
            else:
                target_element.Update(color)
        elif self.BType == BUTTON_TYPE_CLOSES_WIN:
            if self.Key is not None:
                self.ParentForm.LastButtonClicked = self.Key
            else:
                self.ParentForm.LastButtonClicked = self.ButtonText
            self.ParentForm.FormRemainedOpen = False
            if self.ParentForm.CurrentlyRunningMainloop:
                self.ParentForm.App.ExitMainLoop()
            self.ParentForm.IgnoreClose = True
            self.ParentForm.MasterFrame.Close()
            if self.ParentForm.NonBlocking:
                Window._DecrementOpenCount()
            self.ParentForm._Close()
        elif self.BType == BUTTON_TYPE_READ_FORM:
            self.ParentForm.FormRemainedOpen = True
            element_callback_quit_mainloop(self)
        elif self.BType == BUTTON_TYPE_CLOSES_WIN_ONLY:
            element_callback_quit_mainloop(self)
            self.ParentForm._Close()
            Window._DecrementOpenCount()
        elif self.BType == BUTTON_TYPE_CALENDAR_CHOOSER:
            should_submit_window = False
        if should_submit_window:
            self.ParentForm.LastButtonClicked = target_element.Key
            self.ParentForm.FormRemainedOpen = True
            self.ParentForm.MessageQueue.put(self.ParentForm.LastButtonClicked)
        return

    def Update(self, text=None, button_color=(None, None), disabled=None, image_data=None, image_filename=None, font=None, visible=None, image_subsample=None, image_size=(None, None)):
        if False:
            for i in range(10):
                print('nop')
        if text is not None:
            self.Widget.set_text(str(text))
        (fg, bg) = button_color
        if image_data:
            self.Widget.empty()
            simage = SuperImage(image_data)
            if image_size is not (None, None):
                simage.set_size(image_size[0], image_size[1])
            self.Widget.append(simage)
        if image_filename:
            self.Widget.empty()
            simage = SuperImage(image_filename)
            if image_size is not (None, None):
                simage.set_size(image_size[0], image_size[1])
            self.Widget.append(simage)
        super().Update(self.Widget, background_color=bg, text_color=fg, disabled=disabled, font=font, visible=visible)

    def GetText(self):
        if False:
            for i in range(10):
                print('nop')
        return self.Widget.get_text()
    get_text = GetText
    update = Update
B = Button
Btn = Button
Butt = Button

def convert_tkinter_filetypes_to_wx(filetypes):
    if False:
        for i in range(10):
            print('nop')
    wx_filetypes = ''
    for item in filetypes:
        filetype = item[0] + ' (' + item[1] + ')|' + item[1]
        wx_filetypes += filetype
    return wx_filetypes

class ProgressBar(Element):

    def __init__(self, max_value, orientation=None, size=(None, None), auto_size_text=None, bar_color=(None, None), style=None, border_width=None, relief=None, key=None, pad=None):
        if False:
            i = 10
            return i + 15
        '\n        ProgressBar Element\n        :param max_value:\n        :param orientation:\n        :param size:\n        :param auto_size_text:\n        :param bar_color:\n        :param style:\n        :param border_width:\n        :param relief:\n        :param key:\n        :param pad:\n        '
        self.MaxValue = max_value
        self.TKProgressBar = None
        self.Cancelled = False
        self.NotRunning = True
        self.Orientation = orientation if orientation else DEFAULT_METER_ORIENTATION
        self.BarColor = bar_color
        self.BarStyle = style if style else DEFAULT_PROGRESS_BAR_STYLE
        self.BorderWidth = border_width if border_width else DEFAULT_PROGRESS_BAR_BORDER_WIDTH
        self.Relief = relief if relief else DEFAULT_PROGRESS_BAR_RELIEF
        self.BarExpired = False
        super().__init__(ELEM_TYPE_PROGRESS_BAR, size=size, auto_size_text=auto_size_text, key=key, pad=pad)

    def UpdateBar(self, current_count, max=None):
        if False:
            for i in range(10):
                print('nop')
        print('*** NOT IMPLEMENTED ***')
        return
        if self.ParentForm.TKrootDestroyed:
            return False
        self.TKProgressBar.Update(current_count, max=max)
        try:
            self.ParentForm.TKroot.update()
        except:
            _my_windows.Decrement()
            return False
        return True
    update_bar = UpdateBar

class Image(Element):

    def __init__(self, filename=None, data=None, background_color=None, size=(None, None), pad=None, key=None, tooltip=None, right_click_menu=None, visible=True, enable_events=False):
        if False:
            for i in range(10):
                print('nop')
        '\n        Image Element\n        :param filename:\n        :param data:\n        :param background_color:\n        :param size:\n        :param pad:\n        :param key:\n        :param tooltip:\n        '
        self.Filename = filename if filename else None
        self.Data = data
        self.tktext_label = None
        self.BackgroundColor = background_color
        self.Disabled = False
        self.EnableEvents = enable_events
        sz = (0, 0) if size == (None, None) else size
        self.Widget = None
        super().__init__(ELEM_TYPE_IMAGE, size=sz, background_color=background_color, pad=pad, key=key, tooltip=tooltip, visible=visible)
        return

    def Update(self, filename=None, data=None, size=(None, None), visible=None):
        if False:
            while True:
                i = 10
        if data is not None:
            self.Widget.load(data)
        if filename is not None:
            self.Widget.load(filename)
        super().Update(self.Widget, visible=visible)
    update = Update

class SuperImage(remi.gui.Image):

    def __init__(self, file_path_name=None, **kwargs):
        if False:
            i = 10
            return i + 15
        '\n        This new app_instance variable is causing lots of problems.  I do not know the value of the App\n        when I create this image.\n        :param app_instance:\n        :param file_path_name:\n        :param kwargs:\n        '
        image = file_path_name
        super(SuperImage, self).__init__(image, **kwargs)
        self.imagedata = None
        self.mimetype = None
        self.encoding = None
        if not image:
            return
        self.load(image)

    def load(self, file_path_name):
        if False:
            for i in range(10):
                print('nop')
        if type(file_path_name) is bytes:
            try:
                self.imagedata = base64.b64decode(file_path_name, validate=True)
                self.attributes['src'] = '/%s/get_image_data?update_index=%s' % (id(self), str(time.time()))
            except binascii.Error:
                self.imagedata = file_path_name
                self.refresh()
            self.refresh()
        else:
            self.attributes['src'] = remi.gui.load_resource(file_path_name)
            "print(f'***** Loading file = {file_path_name}')\n            self.mimetype, self.encoding = mimetypes.guess_type(file_path_name)\n            with open(file_path_name, 'rb') as f:\n                self.imagedata = f.read()"
            self.refresh()

    def refresh(self):
        if False:
            i = 10
            return i + 15
        i = int(time.time() * 1000000.0)
        if Window.App is not None:
            Window.App.execute_javascript("\n                var url = '/%(id)s/get_image_data?update_index=%(frame_index)s';\n                var xhr = new XMLHttpRequest();\n                xhr.open('GET', url, true);\n                xhr.responseType = 'blob'\n                xhr.onload = function(e){\n                    var urlCreator = window.URL || window.webkitURL;\n                    var imageUrl = urlCreator.createObjectURL(this.response);\n                    document.getElementById('%(id)s').src = imageUrl;\n                }\n                xhr.send();\n                " % {'id': id(self), 'frame_index': i})

    def get_image_data(self, update_index):
        if False:
            print('Hello World!')
        headers = {'Content-type': self.mimetype if self.mimetype else 'application/octet-stream'}
        return [self.imagedata, headers]

class Graph(Element):

    def __init__(self, canvas_size, graph_bottom_left, graph_top_right, background_color=None, pad=None, change_submits=False, drag_submits=False, size_px=(None, None), enable_events=False, key=None, visible=True, disabled=False, tooltip=None):
        if False:
            while True:
                i = 10
        '\n        Graph Element\n        :param canvas_size:\n        :param graph_bottom_left:\n        :param graph_top_right:\n        :param background_color:\n        :param pad:\n        :param key:\n        :param tooltip:\n        '
        self.CanvasSize = canvas_size
        self.BottomLeft = graph_bottom_left
        self.TopRight = graph_top_right
        self.ChangeSubmits = change_submits or enable_events
        self.DragSubmits = drag_submits
        self.ClickPosition = (None, None)
        self.MouseButtonDown = False
        self.Disabled = disabled
        self.Widget = None
        self.SvgGroup = None
        super().__init__(ELEM_TYPE_GRAPH, size=canvas_size, size_px=size_px, visible=visible, background_color=background_color, pad=pad, tooltip=tooltip, key=key)
        return

    def _convert_xy_to_canvas_xy(self, x_in, y_in):
        if False:
            while True:
                i = 10
        if None in (x_in, y_in):
            return (None, None)
        scale_x = (self.CanvasSize[0] - 0) / (self.TopRight[0] - self.BottomLeft[0])
        scale_y = (0 - self.CanvasSize[1]) / (self.TopRight[1] - self.BottomLeft[1])
        new_x = 0 + scale_x * (x_in - self.BottomLeft[0])
        new_y = self.CanvasSize[1] + scale_y * (y_in - self.BottomLeft[1])
        return (new_x, new_y)

    def _convert_canvas_xy_to_xy(self, x_in, y_in):
        if False:
            while True:
                i = 10
        if None in (x_in, y_in):
            return (None, None)
        (x_in, y_in) = (int(x_in), int(y_in))
        scale_x = (self.CanvasSize[0] - 0) / (self.TopRight[0] - self.BottomLeft[0])
        scale_y = (0 - self.CanvasSize[1]) / (self.TopRight[1] - self.BottomLeft[1])
        new_x = x_in / scale_x + self.BottomLeft[0]
        new_y = (y_in - self.CanvasSize[1]) / scale_y + self.BottomLeft[1]
        return (int(new_x), int(new_y))

    def DrawLine(self, point_from, point_to, color='black', width=1):
        if False:
            print('Hello World!')
        if point_from == (None, None) or color is None:
            return
        converted_point_from = self._convert_xy_to_canvas_xy(point_from[0], point_from[1])
        converted_point_to = self._convert_xy_to_canvas_xy(point_to[0], point_to[1])
        if self.Widget is None:
            print('*** WARNING - The Graph element has not been finalized and cannot be drawn upon ***')
            print('Call Window.Finalize() prior to this operation')
            return None
        line = remi.gui.SvgLine(converted_point_from[0], converted_point_from[1], converted_point_to[0], converted_point_to[1])
        line.set_stroke(width, color)
        self.SvgGroup.append([line])
        return line

    def DrawPoint(self, point, size=2, color='black'):
        if False:
            return 10
        if point == (None, None):
            return
        converted_point = self._convert_xy_to_canvas_xy(point[0], point[1])
        if self.Widget is None:
            print('*** WARNING - The Graph element has not been finalized and cannot be drawn upon ***')
            print('Call Window.Finalize() prior to this operation')
            return None
        rpoint = remi.gui.SvgCircle(converted_point[0], converted_point[1], size)
        rpoint.set_stroke(size, color)
        rpoint.set_fill(color)
        self.SvgGroup.append([rpoint])
        return rpoint

    def DrawCircle(self, center_location, radius, fill_color=None, line_color='black'):
        if False:
            i = 10
            return i + 15
        if center_location == (None, None):
            return
        converted_point = self._convert_xy_to_canvas_xy(center_location[0], center_location[1])
        if self.Widget is None:
            print('*** WARNING - The Graph element has not been finalized and cannot be drawn upon ***')
            print('Call Window.Finalize() prior to this operation')
            return None
        rpoint = remi.gui.SvgCircle(converted_point[0], converted_point[1], radius=radius)
        rpoint.set_fill(fill_color)
        rpoint.set_stroke(color=line_color)
        self.SvgGroup.append([rpoint])
        return rpoint

    def DrawOval(self, top_left, bottom_right, fill_color=None, line_color=None):
        if False:
            print('Hello World!')
        converted_top_left = self._convert_xy_to_canvas_xy(top_left[0], top_left[1])
        converted_bottom_right = self._convert_xy_to_canvas_xy(bottom_right[0], bottom_right[1])
        if self.Widget is None:
            print('*** WARNING - The Graph element has not been finalized and cannot be drawn upon ***')
            print('Call Window.Finalize() prior to this operation')
            return None
        return

    def DrawRectangle(self, top_left, bottom_right, fill_color=None, line_color='black'):
        if False:
            print('Hello World!')
        converted_top_left = self._convert_xy_to_canvas_xy(top_left[0], top_left[1])
        converted_bottom_right = self._convert_xy_to_canvas_xy(bottom_right[0], bottom_right[1])
        if self.Widget is None:
            print('*** WARNING - The Graph element has not been finalized and cannot be drawn upon ***')
            print('Call Window.Finalize() prior to this operation')
            return None
        rpoint = remi.gui.SvgRectangle(converted_top_left[0], converted_top_left[1], abs(converted_bottom_right[0] - converted_top_left[0]), abs(converted_top_left[1] - converted_bottom_right[1]))
        rpoint.set_stroke(width=1, color=line_color)
        if fill_color is not None:
            rpoint.set_fill(fill_color)
        else:
            rpoint.set_fill('transparent')
        self.SvgGroup.append([rpoint])
        return rpoint

    def DrawText(self, text, location, color='black', font=None, angle=0):
        if False:
            for i in range(10):
                print('nop')
        text = str(text)
        if location == (None, None):
            return
        converted_point = self._convert_xy_to_canvas_xy(location[0], location[1])
        if self.Widget is None:
            print('*** WARNING - The Graph element has not been finalized and cannot be drawn upon ***')
            print('Call Window.Finalize() prior to this operation')
            return None
        rpoint = remi.gui.SvgText(converted_point[0], converted_point[1], text)
        self.SvgGroup.append([rpoint])
        return rpoint

    def DrawImage(self, data=None, image_source=None, location=(None, None), size=(100, 100)):
        if False:
            while True:
                i = 10
        if location == (None, None):
            return
        if data is not None:
            image_source = data.decode('utf-8') if type(data) is bytes else data
        converted_point = self._convert_xy_to_canvas_xy(location[0], location[1])
        if self.Widget is None:
            print('*** WARNING - The Graph element has not been finalized and cannot be drawn upon ***')
            print('Call Window.Finalize() prior to this operation')
            return None
        rpoint = remi.gui.SvgImage('', converted_point[0], converted_point[0], size[0], size[1])
        if type(image_source) is bytes or len(image_source) > 200:
            rpoint.image_data = 'data:image/svg;base64,%s' % image_source
        else:
            (mimetype, encoding) = mimetypes.guess_type(image_source)
            with open(image_source, 'rb') as f:
                data = f.read()
            b64 = base64.b64encode(data)
            b64_str = b64.decode('utf-8')
            image_string = 'data:image/svg;base64,%s' % b64_str
            rpoint.image_data = image_string
        self.SvgGroup.append([rpoint])
        rpoint.redraw()
        self.SvgGroup.redraw()
        return rpoint

    def Erase(self):
        if False:
            for i in range(10):
                print('nop')
        if self.Widget is None:
            print('*** WARNING - The Graph element has not been finalized and cannot be drawn upon ***')
            print('Call Window.Finalize() prior to this operation')
            return None
        self.Widget.empty()
        self.SvgGroup = remi.gui.SvgSubcontainer(0, 0, '100%', '100%')
        self.Widget.append(self.SvgGroup)

    def Update(self, background_color):
        if False:
            return 10
        if self.Widget is None:
            print('*** WARNING - The Graph element has not been finalized and cannot be drawn upon ***')
            print('Call Window.Finalize() prior to this operation')
            return None
        if self.BackgroundColor not in (None, COLOR_SYSTEM_DEFAULT):
            self.Widget.style['background-color'] = self.BackgroundColor

    def Move(self, x_direction, y_direction):
        if False:
            for i in range(10):
                print('nop')
        zero_converted = self._convert_xy_to_canvas_xy(0, 0)
        shift_converted = self._convert_xy_to_canvas_xy(x_direction, y_direction)
        shift_amount = (shift_converted[0] - zero_converted[0], shift_converted[1] - zero_converted[1])
        if self.Widget is None:
            print('*** WARNING - The Graph element has not been finalized and cannot be drawn upon ***')
            print('Call Window.Finalize() prior to this operation')
            return None
        print(self.SvgGroup.attributes)
        cur_x = float(self.SvgGroup.attributes['x'])
        cur_y = float(self.SvgGroup.attributes['y'])
        self.SvgGroup.set_position(cur_x - x_direction, cur_y - y_direction)
        self.SvgGroup.redraw()

    def Relocate(self, x, y):
        if False:
            while True:
                i = 10
        shift_converted = self._convert_xy_to_canvas_xy(x, y)
        if self.Widget is None:
            print('*** WARNING - Your figure is None. It most likely means your did not Finalize your Window ***')
            print('Call Window.Finalize() prior to all graph operations')
            return None
        self.SvgGroup.set_position(shift_converted[0], shift_converted[1])
        self.SvgGroup.redraw()

    def MoveFigure(self, figure, x_direction, y_direction):
        if False:
            return 10
        figure = figure
        zero_converted = self._convert_xy_to_canvas_xy(0, 0)
        shift_converted = self._convert_xy_to_canvas_xy(x_direction, y_direction)
        shift_amount = (shift_converted[0] - zero_converted[0], shift_converted[1] - zero_converted[1])
        if figure is None:
            print('*** WARNING - Your figure is None. It most likely means your did not Finalize your Window ***')
            print('Call Window.Finalize() prior to all graph operations')
            return None
        print(figure.attributes)
        try:
            cur_x = float(figure.attributes['x'])
            cur_y = float(figure.attributes['y'])
            figure.set_position(cur_x - x_direction, cur_y - y_direction)
        except:
            cur_x1 = float(figure.attributes['x1'])
            cur_x2 = float(figure.attributes['x2'])
            cur_y1 = float(figure.attributes['y1'])
            cur_y2 = float(figure.attributes['y2'])
            figure.set_coords(cur_x1 - x_direction, cur_y1 - y_direction, cur_x2 - x_direction, cur_y2 - x_direction)
        figure.redraw()

    def RelocateFigure(self, figure, x, y):
        if False:
            while True:
                i = 10
        figure = figure
        zero_converted = self._convert_xy_to_canvas_xy(0, 0)
        shift_converted = self._convert_xy_to_canvas_xy(x, y)
        shift_amount = (shift_converted[0] - zero_converted[0], shift_converted[1] - zero_converted[1])
        if figure is None:
            print('*** WARNING - Your figure is None. It most likely means your did not Finalize your Window ***')
            print('Call Window.Finalize() prior to all graph operations')
            return None
        figure.set_position(shift_converted[0], shift_converted[1])
        figure.redraw()

    def DeleteFigure(self, figure):
        if False:
            return 10
        figure = figure
        if figure is None:
            print('*** WARNING - Your figure is None. It most likely means your did not Finalize your Window ***')
            print('Call Window.Finalize() prior to all graph operations')
            return None
        self.SvgGroup.remove_child(figure)
        del figure

    def change_coordinates(self, graph_bottom_left, graph_top_right):
        if False:
            for i in range(10):
                print('nop')
        '\n        Changes the corrdinate system to a new one.  The same 2 points in space are used to define the coorinate\n        system - the bottom left and the top right values of your graph.\n\n        :param graph_bottom_left: Tuple[int, int] (x,y) The bottoms left corner of your coordinate system\n        :param graph_top_right: Tuple[int, int]  (x,y) The top right corner of  your coordinate system\n        '
        self.BottomLeft = graph_bottom_left
        self.TopRight = graph_top_right

    def _MouseDownCallback(self, widget, x, y, *args):
        if False:
            while True:
                i = 10
        self.MouseButtonDown = True

    def _MouseUpCallback(self, widget, x, y, *args):
        if False:
            print('Hello World!')
        self.ClickPosition = self._convert_canvas_xy_to_xy(int(x), int(y))
        self.MouseButtonDown = False
        if self.ChangeSubmits:
            self.ParentForm.LastButtonClicked = self.Key if self.Key is not None else ''
            self.ParentForm.MessageQueue.put(self.ParentForm.LastButtonClicked)

    def ClickCallback(self, widget: remi.gui.Svg, *args):
        if False:
            i = 10
            return i + 15
        return
        self.ClickPosition = (None, None)
        self.ParentForm.LastButtonClicked = self.Key if self.Key is not None else ''
        self.ParentForm.MessageQueue.put(self.ParentForm.LastButtonClicked)

    def _DragCallback(self, emitter, x, y):
        if False:
            i = 10
            return i + 15
        if not self.MouseButtonDown:
            return
        self.ClickPosition = self._convert_canvas_xy_to_xy(x, y)
        self.ParentForm.LastButtonClicked = self.Key if self.Key is not None else ''
        self.ParentForm.MessageQueue.put(self.ParentForm.LastButtonClicked)
    click_callback = ClickCallback
    delete_figure = DeleteFigure
    draw_circle = DrawCircle
    draw_image = DrawImage
    draw_line = DrawLine
    draw_oval = DrawOval
    draw_point = DrawPoint
    draw_rectangle = DrawRectangle
    draw_text = DrawText
    erase = Erase
    move = Move
    move_figure = MoveFigure
    relocate = Relocate
    relocate_figure = RelocateFigure
    update = Update

class CLASSframe(remi.gui.VBox):

    def __init__(self, title, *args, **kwargs):
        if False:
            while True:
                i = 10
        super(CLASSframe, self).__init__(*args, **kwargs)
        self.style.update({'overflow': 'visible', 'border-width': '1px', 'border-style': 'solid', 'border-color': '#7d7d7d'})
        self.frame_label = remi.gui.Label('frame label')
        self.frame_label.style.update({'position': 'relative', 'overflow': 'auto', 'background-color': '#ffffff', 'border-width': '1px', 'border-style': 'solid', 'top': '-7px', 'width': '0px', 'height': '0px', 'left': '10px'})
        self.append(self.frame_label, 'frame_label')
        self.set_title(title)

    def set_title(self, title):
        if False:
            while True:
                i = 10
        self.frame_label.set_text(title)

class Frame(Element):

    def __init__(self, title, layout, title_color=None, background_color=None, title_location=None, relief=DEFAULT_FRAME_RELIEF, element_justification='left', size=(None, None), font=None, pad=None, border_width=None, key=None, tooltip=None):
        if False:
            i = 10
            return i + 15
        '\n        Frame Element\n        :param title:\n        :param layout:\n        :param title_color:\n        :param background_color:\n        :param title_location:\n        :param relief:\n        :param size:\n        :param font:\n        :param pad:\n        :param border_width:\n        :param key:\n        :param tooltip:\n        '
        self.UseDictionary = False
        self.ReturnValues = None
        self.ReturnValuesList = []
        self.ReturnValuesDictionary = {}
        self.DictionaryKeyCounter = 0
        self.ParentWindow = None
        self.Rows = []
        self.TKFrame = None
        self.Title = title
        self.Relief = relief
        self.TitleLocation = title_location
        self.BorderWidth = border_width
        self.BackgroundColor = background_color if background_color is not None else DEFAULT_BACKGROUND_COLOR
        self.Justification = 'left'
        self.ElementJustification = element_justification
        self.Widget = None
        self.Layout(layout)
        super().__init__(ELEM_TYPE_FRAME, background_color=background_color, text_color=title_color, size=size, font=font, pad=pad, key=key, tooltip=tooltip)
        return

    def AddRow(self, *args):
        if False:
            print('Hello World!')
        ' Parms are a variable number of Elements '
        NumRows = len(self.Rows)
        CurrentRowNumber = NumRows
        CurrentRow = []
        for (i, element) in enumerate(args):
            element.Position = (CurrentRowNumber, i)
            element.ParentContainer = self
            CurrentRow.append(element)
            if element.Key is not None:
                self.UseDictionary = True
        self.Rows.append(CurrentRow)

    def Layout(self, rows):
        if False:
            return 10
        for row in rows:
            self.AddRow(*row)

    def _GetElementAtLocation(self, location):
        if False:
            while True:
                i = 10
        (row_num, col_num) = location
        row = self.Rows[row_num]
        element = row[col_num]
        return element
    add_row = AddRow
    layout = Layout

class VerticalSeparator(Element):

    def __init__(self, pad=None):
        if False:
            print('Hello World!')
        '\n        VerticalSeperator - A separator that spans only 1 row in a vertical fashion\n        :param pad:\n        '
        self.Orientation = 'vertical'
        super().__init__(ELEM_TYPE_SEPARATOR, pad=pad)
VSeperator = VerticalSeparator
VSeparator = VerticalSeparator
VSep = VerticalSeparator

class Tab(Element):

    def __init__(self, title, layout, title_color=None, background_color=None, font=None, pad=None, disabled=False, element_justification='left', border_width=None, key=None, tooltip=None):
        if False:
            return 10
        '\n        Tab Element\n        :param title:\n        :param layout:\n        :param title_color:\n        :param background_color:\n        :param font:\n        :param pad:\n        :param disabled:\n        :param border_width:\n        :param key:\n        :param tooltip:\n        '
        self.UseDictionary = False
        self.ReturnValues = None
        self.ReturnValuesList = []
        self.ReturnValuesDictionary = {}
        self.DictionaryKeyCounter = 0
        self.ParentWindow = None
        self.Rows = []
        self.TKFrame = None
        self.Title = title
        self.BorderWidth = border_width
        self.Disabled = disabled
        self.ParentNotebook = None
        self.Justification = 'left'
        self.ElementJustification = element_justification
        self.TabID = None
        self.BackgroundColor = background_color if background_color is not None else DEFAULT_BACKGROUND_COLOR
        self.Widget = None
        self._Layout(layout)
        super().__init__(ELEM_TYPE_TAB, background_color=background_color, text_color=title_color, font=font, pad=pad, key=key, tooltip=tooltip)
        return

    def _AddRow(self, *args):
        if False:
            while True:
                i = 10
        ' Parms are a variable number of Elements '
        NumRows = len(self.Rows)
        CurrentRowNumber = NumRows
        CurrentRow = []
        for (i, element) in enumerate(args):
            element.Position = (CurrentRowNumber, i)
            element.ParentContainer = self
            CurrentRow.append(element)
            if element.Key is not None:
                self.UseDictionary = True
        self.Rows.append(CurrentRow)

    def _Layout(self, rows):
        if False:
            return 10
        for row in rows:
            self._AddRow(*row)
        return self

    def _GetElementAtLocation(self, location):
        if False:
            i = 10
            return i + 15
        (row_num, col_num) = location
        row = self.Rows[row_num]
        element = row[col_num]
        return element

class TabGroup(Element):

    def __init__(self, layout, tab_location=None, title_color=None, selected_title_color=None, background_color=None, font=None, change_submits=False, enable_events=False, pad=None, border_width=None, theme=None, key=None, tooltip=None, visible=True):
        if False:
            print('Hello World!')
        '\n        TabGroup Element\n        :param layout:\n        :param tab_location:\n        :param title_color:\n        :param selected_title_color:\n        :param background_color:\n        :param font:\n        :param change_submits:\n        :param pad:\n        :param border_width:\n        :param theme:\n        :param key:\n        :param tooltip:\n        '
        self.UseDictionary = False
        self.ReturnValues = None
        self.ReturnValuesList = []
        self.ReturnValuesDictionary = {}
        self.DictionaryKeyCounter = 0
        self.ParentWindow = None
        self.SelectedTitleColor = selected_title_color
        self.Rows = []
        self.TKNotebook = None
        self.Widget = None
        self.Justification = 'left'
        self.TabCount = 0
        self.BorderWidth = border_width
        self.Theme = theme
        self.BackgroundColor = background_color if background_color is not None else DEFAULT_BACKGROUND_COLOR
        self.ChangeSubmits = enable_events or change_submits
        self.TabLocation = tab_location
        self.Visible = visible
        self.Disabled = False
        self._Layout(layout)
        super().__init__(ELEM_TYPE_TAB_GROUP, background_color=background_color, text_color=title_color, font=font, pad=pad, key=key, tooltip=tooltip)
        return

    def _AddRow(self, *args):
        if False:
            i = 10
            return i + 15
        ' Parms are a variable number of Elements '
        NumRows = len(self.Rows)
        CurrentRowNumber = NumRows
        CurrentRow = []
        for (i, element) in enumerate(args):
            element.Position = (CurrentRowNumber, i)
            element.ParentContainer = self
            CurrentRow.append(element)
            if element.Key is not None:
                self.UseDictionary = True
        self.Rows.append(CurrentRow)

    def _Layout(self, rows):
        if False:
            return 10
        for row in rows:
            self._AddRow(*row)

    def _GetElementAtLocation(self, location):
        if False:
            print('Hello World!')
        (row_num, col_num) = location
        row = self.Rows[row_num]
        element = row[col_num]
        return element

    def FindKeyFromTabName(self, tab_name):
        if False:
            for i in range(10):
                print('nop')
        for row in self.Rows:
            for element in row:
                if element.Title == tab_name:
                    return element.Key
        return None
    find_key_from_tab_name = FindKeyFromTabName

class Slider(Element):

    def __init__(self, range=(None, None), default_value=None, resolution=None, tick_interval=None, orientation=None, border_width=None, relief=None, change_submits=False, enable_events=False, disabled=False, size=(None, None), font=None, background_color=None, text_color=None, key=None, pad=None, tooltip=None, visible=True, size_px=(None, None)):
        if False:
            while True:
                i = 10
        '\n\n        :param range:\n        :param default_value:\n        :param resolution:\n        :param tick_interval:\n        :param orientation:\n        :param border_width:\n        :param relief:\n        :param change_submits:\n        :param enable_events:\n        :param disabled:\n        :param visible:\n        :param size_px:\n        '
        self.TKScale = None
        self.Range = (1, 10) if range == (None, None) else range
        self.DefaultValue = self.Range[0] if default_value is None else default_value
        self.Orientation = orientation if orientation else DEFAULT_SLIDER_ORIENTATION
        self.BorderWidth = border_width if border_width else DEFAULT_SLIDER_BORDER_WIDTH
        self.Relief = relief if relief else DEFAULT_SLIDER_RELIEF
        self.Resolution = 1 if resolution is None else resolution
        self.ChangeSubmits = change_submits or enable_events
        self.Disabled = disabled
        self.TickInterval = tick_interval
        temp_size = size
        if temp_size == (None, None):
            temp_size = (200, 20) if self.Orientation.startswith('h') else (200, 20)
        elif size[0] is not None and size[0] < 100:
            temp_size = (size[0] * 10, size[1] * 3)
        self.Widget = None
        super().__init__(ELEM_TYPE_INPUT_SLIDER, size=temp_size, font=font, background_color=background_color, text_color=text_color, key=key, pad=pad, tooltip=tooltip, visible=visible, size_px=size_px)
        return

    def Update(self, value=None, range=(None, None), disabled=None, visible=None):
        if False:
            i = 10
            return i + 15
        if value is not None:
            self.Widget.set_value(value)
            self.DefaultValue = value
        if range != (None, None):
            self.Widget.attributes['min'] = '{}'.format(range[0])
            self.Widget.attributes['max'] = '{}'.format(range[1])
        super().Update(self.Widget, disabled=disabled, visible=visible)

    def _SliderCallback(self, widget: remi.Widget, value):
        if False:
            for i in range(10):
                print('nop')
        self.ParentForm.LastButtonClicked = self.Key if self.Key is not None else ''
        self.ParentForm.MessageQueue.put(self.ParentForm.LastButtonClicked)
    update = Update

class Column(Element):

    def __init__(self, layout, background_color=None, size=(None, None), pad=None, scrollable=False, vertical_scroll_only=False, element_justification='left', key=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        Column Element\n        :param layout:\n        :param background_color:\n        :param size:\n        :param pad:\n        :param scrollable:\n        :param key:\n        '
        self.UseDictionary = False
        self.ReturnValues = None
        self.ReturnValuesList = []
        self.ReturnValuesDictionary = {}
        self.DictionaryKeyCounter = 0
        self.ParentWindow = None
        self.Rows = []
        self.TKFrame = None
        self.Scrollable = scrollable
        self.VerticalScrollOnly = vertical_scroll_only
        self.ElementJustification = element_justification
        self.Layout(layout)
        super().__init__(ELEM_TYPE_COLUMN, background_color=background_color, size=size, pad=pad, key=key)
        return

    def AddRow(self, *args):
        if False:
            while True:
                i = 10
        ' Parms are a variable number of Elements '
        NumRows = len(self.Rows)
        CurrentRowNumber = NumRows
        CurrentRow = []
        for (i, element) in enumerate(args):
            element.Position = (CurrentRowNumber, i)
            element.ParentContainer = self
            CurrentRow.append(element)
            if element.Key is not None:
                self.UseDictionary = True
        self.Rows.append(CurrentRow)

    def Layout(self, rows):
        if False:
            print('Hello World!')
        for row in rows:
            self.AddRow(*row)

    def _GetElementAtLocation(self, location):
        if False:
            print('Hello World!')
        (row_num, col_num) = location
        row = self.Rows[row_num]
        element = row[col_num]
        return element
    add_row = AddRow
    layout = Layout
Col = Column

class Menu(Element):

    def __init__(self, menu_definition, background_color=COLOR_SYSTEM_DEFAULT, text_color=None, size=(None, None), tearoff=False, pad=None, key=None, disabled=False, font=None):
        if False:
            i = 10
            return i + 15
        '\n        Menu Element\n        :param menu_definition:\n        :param background_color:\n        :param size:\n        :param tearoff:\n        :param pad:\n        :param key:\n        '
        back_color = background_color if background_color is not None else DEFAULT_BACKGROUND_COLOR
        self.MenuDefinition = menu_definition
        self.TKMenu = None
        self.Tearoff = tearoff
        self.Widget = None
        self.MenuItemChosen = None
        self.Disabled = disabled
        super().__init__(ELEM_TYPE_MENUBAR, background_color=back_color, text_color=text_color, size=size, pad=pad, key=key, font=font)
        return

    def _ChangedCallbackMenu(self, widget, *user_data):
        if False:
            return 10
        widget = widget
        chosen = user_data[0]
        self.MenuItemChosen = chosen
        self.ParentForm.LastButtonClicked = chosen
        self.ParentForm.MessageQueue.put(chosen)

class Table(Element):

    def __init__(self, values, headings=None, visible_column_map=None, col_widths=None, def_col_width=10, auto_size_columns=True, max_col_width=20, select_mode=None, display_row_numbers=False, row_header_text='Row', starting_row_num=0, num_rows=None, row_height=None, font=None, justification='right', text_color=None, background_color=None, alternating_row_color=None, row_colors=None, vertical_scroll_only=True, disabled=False, size=(None, None), change_submits=False, enable_events=False, bind_return_key=False, pad=None, key=None, tooltip=None, right_click_menu=None, visible=True, size_px=(None, None)):
        if False:
            while True:
                i = 10
        '\n        Table\n        :param values:\n        :param headings:\n        :param visible_column_map:\n        :param col_widths:\n        :param def_col_width:\n        :param auto_size_columns:\n        :param max_col_width:\n        :param select_mode:\n        :param display_row_numbers:\n        :param num_rows:\n        :param row_height:\n        :param font:\n        :param justification:\n        :param text_color:\n        :param background_color:\n        :param alternating_row_color:\n        :param size:\n        :param change_submits:\n        :param enable_events:\n        :param bind_return_key:\n        :param pad:\n        :param key:\n        :param tooltip:\n        :param right_click_menu:\n        :param visible:\n        '
        self.Values = values
        self.ColumnHeadings = headings
        self.ColumnsToDisplay = visible_column_map
        self.ColumnWidths = col_widths
        self.MaxColumnWidth = max_col_width
        self.DefaultColumnWidth = def_col_width
        self.AutoSizeColumns = auto_size_columns
        self.BackgroundColor = background_color if background_color is not None else DEFAULT_BACKGROUND_COLOR
        self.TextColor = text_color
        self.Justification = justification
        self.InitialState = None
        self.SelectMode = select_mode
        self.DisplayRowNumbers = display_row_numbers
        self.NumRows = num_rows if num_rows is not None else size[1]
        self.RowHeight = row_height
        self.TKTreeview = None
        self.AlternatingRowColor = alternating_row_color
        self.VerticalScrollOnly = vertical_scroll_only
        self.SelectedRows = []
        self.ChangeSubmits = change_submits or enable_events
        self.BindReturnKey = bind_return_key
        self.StartingRowNumber = starting_row_num
        self.RowHeaderText = row_header_text
        self.RightClickMenu = right_click_menu
        self.RowColors = row_colors
        self.Disabled = disabled
        self.SelectedItem = None
        self.SelectedRow = None
        self.Widget = None
        super().__init__(ELEM_TYPE_TABLE, text_color=text_color, background_color=background_color, font=font, size=size, pad=pad, key=key, tooltip=tooltip, visible=visible, size_px=size_px)
        return

    def Update(self, values=None):
        if False:
            return 10
        print('*** Table Update not yet supported ***')
        return
        if values is not None:
            children = self.TKTreeview.get_children()
            for i in children:
                self.TKTreeview.detach(i)
                self.TKTreeview.delete(i)
            children = self.TKTreeview.get_children()
            for (i, value) in enumerate(values):
                if self.DisplayRowNumbers:
                    value = [i + self.StartingRowNumber] + value
                id = self.TKTreeview.insert('', 'end', text=i, iid=i + 1, values=value, tag=i % 2)
            if self.AlternatingRowColor is not None:
                self.TKTreeview.tag_configure(1, background=self.AlternatingRowColor)
            self.Values = values
            self.SelectedRows = []

    def _on_table_row_click(self, table, row, item):
        if False:
            while True:
                i = 10
        self.SelectedItem = item.get_text()
        index = -1
        for (key, value) in table.children.items():
            if value == row:
                index = table._render_children_list.index(key)
                break
        self.SelectedRow = index
        if self.ChangeSubmits:
            self.ParentForm.LastButtonClicked = self.Key if self.Key is not None else ''
            self.ParentForm.MessageQueue.put(self.ParentForm.LastButtonClicked)
        else:
            self.ParentForm.LastButtonClicked = ''

class Tree(Element):

    def __init__(self, data=None, headings=None, visible_column_map=None, col_widths=None, col0_width=10, def_col_width=10, auto_size_columns=True, max_col_width=20, select_mode=None, show_expanded=False, change_submits=False, font=None, justification='right', text_color=None, background_color=None, num_rows=None, pad=None, key=None, tooltip=None):
        if False:
            return 10
        '\n        Tree Element\n        :param headings:\n        :param visible_column_map:\n        :param col_widths:\n        :param def_col_width:\n        :param auto_size_columns:\n        :param max_col_width:\n        :param select_mode:\n        :param font:\n        :param justification:\n        :param text_color:\n        :param background_color:\n        :param num_rows:\n        :param pad:\n        :param key:\n        :param tooltip:\n        '
        self.TreeData = data
        self.ColumnHeadings = headings
        self.ColumnsToDisplay = visible_column_map
        self.ColumnWidths = col_widths
        self.MaxColumnWidth = max_col_width
        self.DefaultColumnWidth = def_col_width
        self.AutoSizeColumns = auto_size_columns
        self.BackgroundColor = background_color if background_color is not None else DEFAULT_BACKGROUND_COLOR
        self.TextColor = text_color
        self.Justification = justification
        self.InitialState = None
        self.SelectMode = select_mode
        self.ShowExpanded = show_expanded
        self.NumRows = num_rows
        self.Col0Width = col0_width
        self.TKTreeview = None
        self.SelectedRows = []
        self.ChangeSubmits = change_submits
        print('*** Tree Element not yet supported ***')
        super().__init__(ELEM_TYPE_TREE, text_color=text_color, background_color=background_color, font=font, pad=pad, key=key, tooltip=tooltip)

    def add_treeview_data(self, node):
        if False:
            print('Hello World!')
        if node.key != '':
            self.TKTreeview.insert(node.parent, 'end', node.key, text=node.text, values=node.values, open=self.ShowExpanded)
        for node in node.children:
            self.add_treeview_data(node)

    def Update(self, values=None, key=None, value=None, text=None):
        if False:
            for i in range(10):
                print('nop')
        print('*** Tree Element not yet supported ***')
        if values is not None:
            children = self.TKTreeview.get_children()
            for i in children:
                self.TKTreeview.detach(i)
                self.TKTreeview.delete(i)
            children = self.TKTreeview.get_children()
            self.TreeData = values
            self.add_treeview_data(self.TreeData.root_node)
            self.SelectedRows = []
        if key is not None:
            item = self.TKTreeview.item(key)
            if value is not None:
                self.TKTreeview.item(key, values=value)
            if text is not None:
                self.TKTreeview.item(key, text=text)
            item = self.TKTreeview.item(key)
        return self
    update = Update

class TreeData(object):

    class Node(object):

        def __init__(self, parent, key, text, values):
            if False:
                while True:
                    i = 10
            self.parent = parent
            self.children = []
            self.key = key
            self.text = text
            self.values = values

        def _Add(self, node):
            if False:
                for i in range(10):
                    print('nop')
            self.children.append(node)

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.tree_dict = {}
        self.root_node = self.Node('', '', 'root', [])
        self.tree_dict[''] = self.root_node

    def _AddNode(self, key, node):
        if False:
            return 10
        self.tree_dict[key] = node

    def Insert(self, parent, key, text, values):
        if False:
            print('Hello World!')
        node = self.Node(parent, key, text, values)
        self.tree_dict[key] = node
        parent_node = self.tree_dict[parent]
        parent_node._Add(node)

    def __repr__(self):
        if False:
            while True:
                i = 10
        return self._NodeStr(self.root_node, 1)

    def _NodeStr(self, node, level):
        if False:
            for i in range(10):
                print('nop')
        return '\n'.join([str(node.key) + ' : ' + str(node.text)] + [' ' * 4 * level + self._NodeStr(child, level + 1) for child in node.children])
    insert = Insert

class ErrorElement(Element):

    def __init__(self, key=None):
        if False:
            i = 10
            return i + 15
        '\n        Error Element\n        :param key:\n        '
        self.Key = key
        super().__init__(ELEM_TYPE_ERROR, key=key)
        return

    def Update(self, *args, **kwargs):
        if False:
            return 10
        PopupError('Keyword error in Update', 'You need to stop this madness and check your spelling', 'Bad key = {}'.format(self.Key), 'Your bad line of code may resemble this:', 'window.FindElement("{}")'.format(self.Key))
        return self

    def Get(self):
        if False:
            while True:
                i = 10
        return 'This is NOT a valid Element!\nSTOP trying to do things with it or I will have to crash at some point!'
    get = Get
    update = Update

class Window:
    _NumOpenWindows = 0
    user_defined_icon = None
    hidden_master_root = None
    QTApplication = None
    active_popups = {}
    highest_level_app = None
    stdout_is_rerouted = False
    stdout_string_io = None
    stdout_location = None
    port_number = 6900
    active_windows = []
    App = None

    def __init__(self, title, layout=None, default_element_size=DEFAULT_ELEMENT_SIZE, default_button_element_size=(None, None), auto_size_text=None, auto_size_buttons=None, location=(None, None), size=(None, None), element_padding=None, button_color=None, font=None, progress_bar_color=(None, None), background_color=None, border_depth=None, auto_close=False, auto_close_duration=None, icon=DEFAULT_BASE64_ICON, force_toplevel=False, alpha_channel=1, return_keyboard_events=False, return_key_down_events=False, use_default_focus=True, text_justification=None, no_titlebar=False, grab_anywhere=False, keep_on_top=False, resizable=True, disable_close=False, margins=(None, None), element_justification='left', disable_minimize=False, background_image=None, finalize=False, web_debug=False, web_ip='0.0.0.0', web_port=0, web_start_browser=True, web_update_interval=1e-07, web_multiple_instance=False):
        if False:
            print('Hello World!')
        '\n\n        :param title:\n        :param default_element_size:\n        :param default_button_element_size:\n        :param auto_size_text:\n        :param auto_size_buttons:\n        :param location:\n        :param size:\n        :param element_padding:\n        :param button_color:\n        :param font:\n        :param progress_bar_color:\n        :param background_color:\n        :param border_depth:\n        :param auto_close:\n        :param auto_close_duration:\n        :param icon:\n        :param force_toplevel:\n        :param alpha_channel:\n        :param return_keyboard_events:\n        :param use_default_focus:\n        :param text_justification:\n        :param no_titlebar:\n        :param grab_anywhere:\n        :param keep_on_top:\n        :param resizable:\n        :param disable_close:\n        :param background_image:\n        '
        self.AutoSizeText = auto_size_text if auto_size_text is not None else DEFAULT_AUTOSIZE_TEXT
        self.AutoSizeButtons = auto_size_buttons if auto_size_buttons is not None else DEFAULT_AUTOSIZE_BUTTONS
        self.Title = title
        self.Rows = []
        self.DefaultElementSize = convert_tkinter_size_to_Wx(default_element_size)
        self.DefaultButtonElementSize = convert_tkinter_size_to_Wx(default_button_element_size) if default_button_element_size != (None, None) else DEFAULT_BUTTON_ELEMENT_SIZE
        self.Location = location
        self.ButtonColor = button_color if button_color else DEFAULT_BUTTON_COLOR
        self.BackgroundColor = background_color if background_color else DEFAULT_BACKGROUND_COLOR
        self.ParentWindow = None
        self.Font = font if font else DEFAULT_FONT
        self.RadioDict = {}
        self.BorderDepth = border_depth
        self.WindowIcon = icon if icon is not None else Window.user_defined_icon
        self.AutoClose = auto_close
        self.NonBlocking = False
        self.TKroot = None
        self.TKrootDestroyed = False
        self.CurrentlyRunningMainloop = False
        self.FormRemainedOpen = False
        self.TKAfterID = None
        self.ProgressBarColor = progress_bar_color
        self.AutoCloseDuration = auto_close_duration
        self.RootNeedsDestroying = False
        self.Shown = False
        self.ReturnValues = None
        self.ReturnValuesList = []
        self.ReturnValuesDictionary = {}
        self.DictionaryKeyCounter = 0
        self.AllKeysDict = {}
        self.LastButtonClicked = None
        self.LastButtonClickedWasRealtime = False
        self.UseDictionary = False
        self.UseDefaultFocus = use_default_focus
        self.ReturnKeyboardEvents = return_keyboard_events
        self.ReturnKeyDownEvents = return_key_down_events
        self.KeyInfoDict = {}
        self.LastKeyboardEvent = None
        self.TextJustification = text_justification
        self.NoTitleBar = no_titlebar
        self.GrabAnywhere = grab_anywhere
        self.KeepOnTop = keep_on_top
        self.ForcefTopLevel = force_toplevel
        self.Resizable = resizable
        self._AlphaChannel = alpha_channel
        self.Timeout = None
        self.TimeoutKey = TIMEOUT_KEY
        self.TimerCancelled = False
        self.DisableClose = disable_close
        self._Hidden = False
        self._Size = size
        self.ElementPadding = element_padding or DEFAULT_ELEMENT_PADDING
        self.FocusElement = None
        self.BackgroundImage = background_image
        self.XFound = False
        self.DisableMinimize = disable_minimize
        self.OutputElementForStdOut = None
        self.Justification = 'left'
        self.ElementJustification = element_justification
        self.IgnoreClose = False
        self.thread_id = None
        self.App = None
        self.web_debug = web_debug
        self.web_ip = web_ip
        self.web_port = web_port
        self.web_start_browser = web_start_browser
        self.web_update_interval = web_update_interval
        self.web_multiple_instance = web_multiple_instance
        self.MessageQueue = Queue()
        self.master_widget = None
        self.UniqueKeyCounter = 0
        if layout is not None:
            self.Layout(layout)
            if finalize:
                self.Finalize()

    @classmethod
    def IncrementOpenCount(self):
        if False:
            return 10
        self._NumOpenWindows += 1

    @classmethod
    def _DecrementOpenCount(self):
        if False:
            i = 10
            return i + 15
        self._NumOpenWindows -= 1 * (self._NumOpenWindows != 0)

    def AddRow(self, *args):
        if False:
            while True:
                i = 10
        ' Parms are a variable number of Elements '
        NumRows = len(self.Rows)
        CurrentRowNumber = NumRows
        CurrentRow = []
        for (i, element) in enumerate(args):
            element.Position = (CurrentRowNumber, i)
            element.ParentContainer = self
            CurrentRow.append(element)
        self.Rows.append(CurrentRow)

    def AddRows(self, rows):
        if False:
            i = 10
            return i + 15
        for row in rows:
            self.AddRow(*row)

    def Layout(self, rows):
        if False:
            while True:
                i = 10
        self.AddRows(rows)
        self._BuildKeyDict()
        return self

    def LayoutAndRead(self, rows, non_blocking=False):
        if False:
            print('Hello World!')
        raise DeprecationWarning('LayoutAndRead is no longer supported... change your call to window.Layout(layout).Read()')

    def LayoutAndShow(self, rows):
        if False:
            for i in range(10):
                print('nop')
        raise DeprecationWarning('LayoutAndShow is no longer supported... change your call to LayoutAndRead')

    def Show(self, non_blocking=False):
        if False:
            i = 10
            return i + 15
        self.Shown = True
        self.NumRows = len(self.Rows)
        self.NumCols = max((len(row) for row in self.Rows))
        self.NonBlocking = non_blocking
        found_focus = False
        for row in self.Rows:
            for element in row:
                try:
                    if element.Focus:
                        found_focus = True
                except:
                    pass
                try:
                    if element.Key is not None:
                        self.UseDictionary = True
                except:
                    pass
        if not found_focus and self.UseDefaultFocus:
            self.UseDefaultFocus = True
        else:
            self.UseDefaultFocus = False
        StartupTK(self)

    def Read(self, timeout=None, timeout_key=TIMEOUT_KEY, close=False):
        if False:
            while True:
                i = 10
        '\n        THE biggest deal method in the Window class! This is how you get all of your data from your Window.\n            Pass in a timeout (in milliseconds) to wait for a maximum of timeout milliseconds. Will return timeout_key\n            if no other GUI events happen first.\n        Use the close parameter to close the window after reading\n\n        :param timeout: (int) Milliseconds to wait until the Read will return IF no other GUI events happen first\n        :param timeout_key: (Any) The value that will be returned from the call if the timer expired\n        :param close: (bool) if True the window will be closed prior to returning\n        :return: Tuple[(Any), Union[Dict[Any:Any]], List[Any], None] (event, values)\n        '
        results = self._read(timeout=timeout, timeout_key=timeout_key)
        if close:
            self.close()
        return results

    def _read(self, timeout=None, timeout_key=TIMEOUT_KEY):
        if False:
            return 10
        self.Timeout = timeout
        self.TimeoutKey = timeout_key
        self.NonBlocking = False
        if not self.Shown:
            self.Show()
        if self.LastButtonClicked is not None and (not self.LastButtonClickedWasRealtime):
            results = BuildResults(self, False, self)
            self.LastButtonClicked = None
            return results
        InitializeResults(self)
        if self.LastButtonClickedWasRealtime:
            try:
                rc = self.TKroot.update()
            except:
                self.TKrootDestroyed = True
                Window._DecrementOpenCount()
            results = BuildResults(self, False, self)
            if results[0] != None and results[0] != timeout_key:
                return results
            else:
                pass
        self.CurrentlyRunningMainloop = True
        if timeout is not None:
            try:
                self.LastButtonClicked = self.MessageQueue.get(timeout=(timeout if timeout else 0.001) / 1000)
            except:
                self.LastButtonClicked = timeout_key
        else:
            self.LastButtonClicked = self.MessageQueue.get()
        results = BuildResults(self, False, self)
        return results

    def _ReadNonBlocking(self):
        if False:
            return 10
        if self.TKrootDestroyed:
            return (None, None)
        if not self.Shown:
            self.Show(non_blocking=True)
        timer = wx.Timer(self.App)
        self.App.Bind(wx.EVT_TIMER, self.timer_timeout)
        timer.Start(milliseconds=0, oneShot=wx.TIMER_ONE_SHOT)
        self.CurrentlyRunningMainloop = True
        self.App.MainLoop()
        if Window.stdout_is_rerouted:
            sys.stdout = Window.stdout_location
        self.CurrentlyRunningMainloop = False
        timer.Stop()
        return BuildResults(self, False, self)

    def SetIcon(self, icon=None, pngbase64=None):
        if False:
            print('Hello World!')
        pass

    def _GetElementAtLocation(self, location):
        if False:
            i = 10
            return i + 15
        (row_num, col_num) = location
        row = self.Rows[row_num]
        element = row[col_num]
        return element

    def _GetDefaultElementSize(self):
        if False:
            while True:
                i = 10
        return self.DefaultElementSize

    def _AutoCloseAlarmCallback(self):
        if False:
            print('Hello World!')
        try:
            window = self
            if window:
                if window.NonBlocking:
                    self.CloseNonBlockingForm()
                else:
                    window._Close()
                    if self.CurrentlyRunningMainloop:
                        self.QTApplication.exit()
                    self.RootNeedsDestroying = True
                    self.QT_QMainWindow.close()
        except:
            pass

    def timer_timeout(self, event):
        if False:
            i = 10
            return i + 15
        if self.TimerCancelled:
            return
        self.LastButtonClicked = self.TimeoutKey
        self.FormRemainedOpen = True
        if self.CurrentlyRunningMainloop:
            self.App.ExitMainLoop()

    def non_block_timer_timeout(self, event):
        if False:
            print('Hello World!')
        self.App.ExitMainLoop()

    def autoclose_timer_callback(self, frame):
        if False:
            i = 10
            return i + 15
        try:
            frame.Close()
        except:
            pass
        if self.CurrentlyRunningMainloop:
            self.App.ExitMainLoop()

    def on_key_down(self, emitter, key, keycode, ctrl, shift, alt):
        if False:
            return 10
        self.LastButtonClicked = 'DOWN' + key
        self.MessageQueue.put(self.LastButtonClicked)
        self.KeyInfoDict = {'key': key, 'keycode': keycode, 'ctrl': ctrl, 'shift': shift, 'alt': alt}

    def on_key_up(self, emitter, key, keycode, ctrl, shift, alt):
        if False:
            print('Hello World!')
        self.LastButtonClicked = key
        self.MessageQueue.put(self.LastButtonClicked)
        self.KeyInfoDict = {'key': key, 'keycode': keycode, 'ctrl': ctrl, 'shift': shift, 'alt': alt}

    def callback_keyboard_char(self, event):
        if False:
            return 10
        self.LastButtonClicked = None
        self.FormRemainedOpen = True
        if event.ClassName == 'wxMouseEvent':
            if event.WheelRotation < 0:
                self.LastKeyboardEvent = 'MouseWheel:Down'
            else:
                self.LastKeyboardEvent = 'MouseWheel:Up'
        else:
            self.LastKeyboardEvent = event.GetKeyCode()
        if not self.NonBlocking:
            BuildResults(self, False, self)
        if self.CurrentlyRunningMainloop:
            self.App.ExitMainLoop()
        if event.ClassName != 'wxMouseEvent':
            event.DoAllowNextEvent()

    def Finalize(self):
        if False:
            print('Hello World!')
        if self.TKrootDestroyed:
            return self
        if not self.Shown:
            self.Show(non_blocking=True)
        return self

    def Refresh(self):
        if False:
            for i in range(10):
                print('nop')
        return self

    def VisibilityChanged(self):
        if False:
            for i in range(10):
                print('nop')
        self.SizeChanged()
        return self

    def Fill(self, values_dict):
        if False:
            while True:
                i = 10
        _FillFormWithValues(self, values_dict)
        return self

    def FindElement(self, key, silent_on_error=False):
        if False:
            return 10
        try:
            element = self.AllKeysDict[key]
        except KeyError:
            element = None
        if element is None:
            if not silent_on_error:
                print("*** WARNING = FindElement did not find the key. Please check your key's spelling ***")
                PopupError('Keyword error in FindElement Call', 'Bad key = {}'.format(key), 'Your bad line of code may resemble this:', 'window.FindElement("{}")'.format(key))
                return ErrorElement(key=key)
            else:
                return False
        return element
    Element = FindElement

    def _BuildKeyDict(self):
        if False:
            for i in range(10):
                print('nop')
        dict = {}
        self.AllKeysDict = self._BuildKeyDictForWindow(self, self, dict)

    def _BuildKeyDictForWindow(self, top_window, window, key_dict):
        if False:
            i = 10
            return i + 15
        for (row_num, row) in enumerate(window.Rows):
            for (col_num, element) in enumerate(row):
                if element.Type == ELEM_TYPE_COLUMN:
                    key_dict = self._BuildKeyDictForWindow(top_window, element, key_dict)
                if element.Type == ELEM_TYPE_FRAME:
                    key_dict = self._BuildKeyDictForWindow(top_window, element, key_dict)
                if element.Type == ELEM_TYPE_TAB_GROUP:
                    key_dict = self._BuildKeyDictForWindow(top_window, element, key_dict)
                if element.Type == ELEM_TYPE_TAB:
                    key_dict = self._BuildKeyDictForWindow(top_window, element, key_dict)
                if element.Key is None:
                    if element.Type == ELEM_TYPE_BUTTON:
                        element.Key = element.ButtonText
                    if element.Type in (ELEM_TYPE_MENUBAR, ELEM_TYPE_BUTTONMENU, ELEM_TYPE_CANVAS, ELEM_TYPE_INPUT_SLIDER, ELEM_TYPE_GRAPH, ELEM_TYPE_IMAGE, ELEM_TYPE_INPUT_CHECKBOX, ELEM_TYPE_INPUT_LISTBOX, ELEM_TYPE_INPUT_COMBO, ELEM_TYPE_INPUT_MULTILINE, ELEM_TYPE_INPUT_OPTION_MENU, ELEM_TYPE_INPUT_SPIN, ELEM_TYPE_TABLE, ELEM_TYPE_TREE, ELEM_TYPE_INPUT_TEXT):
                        element.Key = top_window.DictionaryKeyCounter
                        top_window.DictionaryKeyCounter += 1
                if element.Key is not None:
                    if element.Key in key_dict.keys():
                        print('*** Duplicate key found in your layout {} ***'.format(element.Key)) if element.Type != ELEM_TYPE_BUTTON else None
                        element.Key = element.Key + str(self.UniqueKeyCounter)
                        self.UniqueKeyCounter += 1
                        print('*** Replaced new key with {} ***'.format(element.Key)) if element.Type != ELEM_TYPE_BUTTON else None
                    key_dict[element.Key] = element
        return key_dict

    def FindElementWithFocus(self):
        if False:
            for i in range(10):
                print('nop')
        return self.FocusElement

    def SaveToDisk(self, filename):
        if False:
            i = 10
            return i + 15
        try:
            results = BuildResults(self, False, self)
            with open(filename, 'wb') as sf:
                pickle.dump(results[1], sf)
        except:
            print('*** Error saving form to disk ***')

    def LoadFromDisk(self, filename):
        if False:
            for i in range(10):
                print('nop')
        try:
            with open(filename, 'rb') as df:
                self.Fill(pickle.load(df))
        except:
            print('*** Error loading form to disk ***')

    def GetScreenDimensions(self):
        if False:
            print('Hello World!')
        size = (0, 0)
        return size

    def Move(self, x, y):
        if False:
            return 10
        self.MasterFrame.SetPosition((x, y))

    def Minimize(self):
        if False:
            for i in range(10):
                print('nop')
        self.MasterFrame.Iconize()

    def Maximize(self):
        if False:
            while True:
                i = 10
        self.MasterFrame.Maximize()

    def _Close(self):
        if False:
            return 10
        if not self.NonBlocking:
            BuildResults(self, False, self)
        if self.TKrootDestroyed:
            return None
        self.TKrootDestroyed = True
        self.RootNeedsDestroying = True
        self.Close()

    def Close(self):
        if False:
            print('Hello World!')
        if len(Window.active_windows) != 0:
            del Window.active_windows[-1]
            if len(Window.active_windows) != 0:
                window = Window.active_windows[-1]
                Window.App.set_root_widget(window.master_widget)
            else:
                self.App.close()
                self.App.server.server_starter_instance._alive = False
                self.App.server.server_starter_instance._sserver.shutdown()
            return
        self.App.close()
        self.App.server.server_starter_instance._alive = False
        self.App.server.server_starter_instance._sserver.shutdown()
    CloseNonBlockingForm = Close
    CloseNonBlocking = Close

    def Disable(self):
        if False:
            while True:
                i = 10
        self.MasterFrame.Enable(False)

    def Enable(self):
        if False:
            for i in range(10):
                print('nop')
        self.MasterFrame.Enable(True)

    def Hide(self):
        if False:
            while True:
                i = 10
        self._Hidden = True
        self.master_widget.attributes['hidden'] = 'true'
        return

    def UnHide(self):
        if False:
            for i in range(10):
                print('nop')
        if self._Hidden:
            del self.master_widget.attributes['hidden']
            self._Hidden = False

    def Disappear(self):
        if False:
            while True:
                i = 10
        self.MasterFrame.SetTransparent(0)

    def Reappear(self):
        if False:
            for i in range(10):
                print('nop')
        self.MasterFrame.SetTransparent(255)

    def SetAlpha(self, alpha):
        if False:
            for i in range(10):
                print('nop')
        "\n        Change the window's transparency\n        :param alpha: From 0 to 1 with 0 being completely transparent\n        :return:\n        "
        self._AlphaChannel = alpha * 255
        if self._AlphaChannel is not None:
            self.MasterFrame.SetTransparent(self._AlphaChannel)

    @property
    def AlphaChannel(self):
        if False:
            i = 10
            return i + 15
        return self._AlphaChannel

    @AlphaChannel.setter
    def AlphaChannel(self, alpha):
        if False:
            i = 10
            return i + 15
        self.SetAlpha(alpha)

    def BringToFront(self):
        if False:
            while True:
                i = 10
        self.MasterFrame.ToggleWindowStyle(wx.STAY_ON_TOP)

    def CurrentLocation(self):
        if False:
            for i in range(10):
                print('nop')
        location = self.MasterFrame.GetPosition()
        return location

    @property
    def Size(self):
        if False:
            return 10
        size = self.MasterFrame.GetSize()
        return size

    @Size.setter
    def Size(self, size):
        if False:
            return 10
        self.MasterFrame.SetSize(size[0], size[1])

    def SizeChanged(self):
        if False:
            print('Hello World!')
        size = self.Size
        self.Size = (size[0] + 1, size[1] + 1)
        self.Size = size
        self.MasterFrame.SetSizer(self.OuterSizer)
        self.OuterSizer.Fit(self.MasterFrame)

    def __getitem__(self, key):
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns Element that matches the passed in key.\n        This is "called" by writing code as thus:\n        window[\'element key\'].Update\n\n        :param key: (Any) The key to find\n        :return: Union[Element, None] The element found or None if no element was found\n        '
        try:
            return self.Element(key)
        except Exception as e:
            print('The key you passed in is no good. Key = {}*'.format(key))
            return None

    def __call__(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        '\n        Call window.Read but without having to type it out.\n        window() == window.Read()\n        window(timeout=50) == window.Read(timeout=50)\n\n        :param args:\n        :param kwargs:\n        :return: Tuple[Any, Dict[Any:Any]] The famous event, values that Read returns.\n        '
        return self.Read(*args, **kwargs)
    add_row = AddRow
    add_rows = AddRows
    alpha_channel = AlphaChannel
    bring_to_front = BringToFront
    close = Close
    current_location = CurrentLocation
    disable = Disable
    disappear = Disappear
    element = Element
    enable = Enable
    fill = Fill
    finalize = Finalize
    find_element = FindElement
    find_element_with_focus = FindElementWithFocus
    get_screen_dimensions = GetScreenDimensions
    hide = Hide
    increment_open_count = IncrementOpenCount
    layout = Layout
    layout_and_read = LayoutAndRead
    layout_and_show = LayoutAndShow
    load_from_disk = LoadFromDisk
    maximize = Maximize
    minimize = Minimize
    move = Move
    num_open_windows = _NumOpenWindows
    read = Read
    reappear = Reappear
    refresh = Refresh
    save_to_disk = SaveToDisk
    set_alpha = SetAlpha
    set_icon = SetIcon
    show = Show
    size = Size
    size_changed = SizeChanged
    un_hide = UnHide
    visibility_changed = VisibilityChanged

    def remi_thread(self):
        if False:
            print('Hello World!')
        logging.getLogger('remi').setLevel(logging.CRITICAL)
        logging.getLogger('remi').disabled = True
        logging.getLogger('remi.server.ws').disabled = True
        logging.getLogger('remi.server').disabled = True
        logging.getLogger('remi.request').disabled = True
        Window.port_number += 1
        try:
            remi.start(self.MyApp, title=self.Title, debug=self.web_debug, address=self.web_ip, port=self.web_port, multiple_instance=self.web_multiple_instance, start_browser=self.web_start_browser, update_interval=self.web_update_interval, userdata=(self,))
        except:
            print('*** ERROR Caught inside Remi ***')
            print(traceback.format_exc())
        print('Returned from Remi Start command... now sending None event')
        self.MessageQueue.put(None)

    class MyApp(remi.App):

        def __init__(self, *args, userdata2=None):
            if False:
                while True:
                    i = 10
            if userdata2 is None:
                userdata = args[-1].userdata
                self.window = userdata[0]
            else:
                self.window = userdata2
            self.master_widget = None
            self.lines_shown = []
            if userdata2 is None:
                super(Window.MyApp, self).__init__(*args, static_file_path={'C': 'c:', 'c': 'c:', 'D': 'd:', 'd': 'd:', 'E': 'e:', 'e': 'e:', 'dot': '.', '.': '.'})

        def _instance(self):
            if False:
                print('Hello World!')
            remi.App._instance(self)
            self.window.App = remi.server.clients[self.session]

        def log_message(self, *args, **kwargs):
            if False:
                return 10
            pass

        def idle(self):
            if False:
                i = 10
                return i + 15
            if Window.stdout_is_rerouted:
                Window.stdout_string_io.seek(0)
                lines = Window.stdout_string_io.readlines()
                if lines != self.lines_shown:
                    self.window.OutputElementForStdOut.Update(''.join(lines))
                self.lines_shown = lines

        def main(self, name='world'):
            if False:
                while True:
                    i = 10
            self.master_widget = setup_remi_window(self, self.window)
            self.window.master_widget = self.master_widget
            self.window.MessageQueue.put('Layout complete')
            return self.master_widget

        def on_window_close(self):
            if False:
                print('Hello World!')
            print('app closing')
            self.close()
            self.server.server_starter_instance._alive = False
            self.server.server_starter_instance._sserver.shutdown()
            print('server stopped')
FlexForm = Window

def element_callback_quit_mainloop(element):
    if False:
        print('Hello World!')
    if element.Key is not None:
        element.ParentForm.LastButtonClicked = element.Key
    else:
        element.ParentForm.LastButtonClicked = ''
    try:
        element.ParentForm.LastButtonClicked = element.Key if element.Key is not None else element.ButtonText
    except:
        element.ParentForm.LastButtonClicked = element.Key
    element.ParentForm.MessageQueue.put(element.ParentForm.LastButtonClicked)

def quit_mainloop(window):
    if False:
        i = 10
        return i + 15
    window.App.ExitMainLoop()

def convert_tkinter_size_to_Wx(size):
    if False:
        return 10
    '\n    Converts size in characters to size in pixels\n    :param size:  size in characters, rows\n    :return: size in pixels, pixels\n    '
    qtsize = size
    if size[1] is not None and size[1] < DEFAULT_PIXEL_TO_CHARS_CUTOFF:
        qtsize = (size[0] * DEFAULT_PIXELS_TO_CHARS_SCALING[0], size[1] * DEFAULT_PIXELS_TO_CHARS_SCALING[1])
    return qtsize

def base64_to_style_image(base64_image):
    if False:
        while True:
            i = 10
    x = "url('data:image/png;base64,"
    x += str(base64_image)
    x += "')"
    return x

def font_parse_string(font):
    if False:
        i = 10
        return i + 15
    '\n    Convert from font string/tyuple into a Qt style sheet string\n    :param font: "Arial 10 Bold" or (\'Arial\', 10, \'Bold)\n    :return: style string that can be combined with other style strings\n    '
    if font is None:
        return ''
    if type(font) is str:
        _font = font.split(' ')
    else:
        _font = font
    family = _font[0]
    point_size = int(_font[1])
    style = _font[2:] if len(_font) > 1 else None
    return (family, point_size, style)

def FolderBrowse(button_text='Browse', target=(ThisRow, -1), initial_folder=None, tooltip=None, size=(None, None), auto_size_button=None, button_color=None, disabled=False, change_submits=False, font=None, pad=None, key=None):
    if False:
        for i in range(10):
            print('nop')
    return Button(button_text=button_text, button_type=BUTTON_TYPE_BROWSE_FOLDER, target=target, initial_folder=initial_folder, tooltip=tooltip, size=size, auto_size_button=auto_size_button, disabled=disabled, button_color=button_color, change_submits=change_submits, font=font, pad=pad, key=key)

def FileBrowse(button_text='Browse', target=(ThisRow, -1), file_types=(('ALL Files', '*.*'),), initial_folder=None, tooltip=None, size=(None, None), auto_size_button=None, button_color=None, change_submits=False, font=None, disabled=False, pad=None, key=None):
    if False:
        print('Hello World!')
    return Button(button_text=button_text, button_type=BUTTON_TYPE_BROWSE_FILE, target=target, file_types=file_types, initial_folder=initial_folder, tooltip=tooltip, size=size, auto_size_button=auto_size_button, change_submits=change_submits, disabled=disabled, button_color=button_color, font=font, pad=pad, key=key)

def FilesBrowse(button_text='Browse', target=(ThisRow, -1), file_types=(('ALL Files', '*.*'),), disabled=False, initial_folder=None, tooltip=None, size=(None, None), auto_size_button=None, button_color=None, change_submits=False, font=None, pad=None, key=None):
    if False:
        while True:
            i = 10
    return Button(button_text=button_text, button_type=BUTTON_TYPE_BROWSE_FILES, target=target, file_types=file_types, initial_folder=initial_folder, change_submits=change_submits, tooltip=tooltip, size=size, auto_size_button=auto_size_button, disabled=disabled, button_color=button_color, font=font, pad=pad, key=key)

def FileSaveAs(button_text='Save As...', target=(ThisRow, -1), file_types=(('ALL Files', '*.*'),), initial_folder=None, disabled=False, tooltip=None, size=(None, None), auto_size_button=None, button_color=None, change_submits=False, font=None, pad=None, key=None):
    if False:
        while True:
            i = 10
    return Button(button_text=button_text, button_type=BUTTON_TYPE_SAVEAS_FILE, target=target, file_types=file_types, initial_folder=initial_folder, tooltip=tooltip, size=size, disabled=disabled, auto_size_button=auto_size_button, button_color=button_color, change_submits=change_submits, font=font, pad=pad, key=key)

def SaveAs(button_text='Save As...', target=(ThisRow, -1), file_types=(('ALL Files', '*.*'),), initial_folder=None, disabled=False, tooltip=None, size=(None, None), auto_size_button=None, button_color=None, change_submits=False, font=None, pad=None, key=None):
    if False:
        i = 10
        return i + 15
    return Button(button_text=button_text, button_type=BUTTON_TYPE_SAVEAS_FILE, target=target, file_types=file_types, initial_folder=initial_folder, tooltip=tooltip, size=size, disabled=disabled, auto_size_button=auto_size_button, button_color=button_color, change_submits=change_submits, font=font, pad=pad, key=key)

def Save(button_text='Save', size=(None, None), auto_size_button=None, button_color=None, bind_return_key=True, disabled=False, tooltip=None, font=None, focus=False, pad=None, key=None):
    if False:
        return 10
    return Button(button_text=button_text, button_type=BUTTON_TYPE_READ_FORM, tooltip=tooltip, size=size, auto_size_button=auto_size_button, button_color=button_color, font=font, disabled=disabled, bind_return_key=bind_return_key, focus=focus, pad=pad, key=key)

def Submit(button_text='Submit', size=(None, None), auto_size_button=None, button_color=None, disabled=False, bind_return_key=True, tooltip=None, font=None, focus=False, pad=None, key=None):
    if False:
        return 10
    return Button(button_text=button_text, button_type=BUTTON_TYPE_READ_FORM, tooltip=tooltip, size=size, auto_size_button=auto_size_button, button_color=button_color, font=font, disabled=disabled, bind_return_key=bind_return_key, focus=focus, pad=pad, key=key)

def Open(button_text='Open', size=(None, None), auto_size_button=None, button_color=None, disabled=False, bind_return_key=True, tooltip=None, font=None, focus=False, pad=None, key=None):
    if False:
        while True:
            i = 10
    return Button(button_text=button_text, button_type=BUTTON_TYPE_READ_FORM, tooltip=tooltip, size=size, auto_size_button=auto_size_button, button_color=button_color, font=font, disabled=disabled, bind_return_key=bind_return_key, focus=focus, pad=pad, key=key)

def OK(button_text='OK', size=(None, None), auto_size_button=None, button_color=None, disabled=False, bind_return_key=True, tooltip=None, font=None, focus=False, pad=None, key=None):
    if False:
        return 10
    return Button(button_text=button_text, button_type=BUTTON_TYPE_READ_FORM, tooltip=tooltip, size=size, auto_size_button=auto_size_button, button_color=button_color, font=font, disabled=disabled, bind_return_key=bind_return_key, focus=focus, pad=pad, key=key)

def Ok(button_text='Ok', size=(None, None), auto_size_button=None, button_color=None, disabled=False, bind_return_key=True, tooltip=None, font=None, focus=False, pad=None, key=None):
    if False:
        i = 10
        return i + 15
    return Button(button_text=button_text, button_type=BUTTON_TYPE_READ_FORM, tooltip=tooltip, size=size, auto_size_button=auto_size_button, button_color=button_color, font=font, disabled=disabled, bind_return_key=bind_return_key, focus=focus, pad=pad, key=key)

def Cancel(button_text='Cancel', size=(None, None), auto_size_button=None, button_color=None, disabled=False, tooltip=None, font=None, bind_return_key=False, focus=False, pad=None, key=None):
    if False:
        for i in range(10):
            print('nop')
    return Button(button_text=button_text, button_type=BUTTON_TYPE_READ_FORM, tooltip=tooltip, size=size, auto_size_button=auto_size_button, button_color=button_color, font=font, disabled=disabled, bind_return_key=bind_return_key, focus=focus, pad=pad, key=key)

def Quit(button_text='Quit', size=(None, None), auto_size_button=None, button_color=None, disabled=False, tooltip=None, font=None, bind_return_key=False, focus=False, pad=None, key=None):
    if False:
        i = 10
        return i + 15
    return Button(button_text=button_text, button_type=BUTTON_TYPE_READ_FORM, tooltip=tooltip, size=size, auto_size_button=auto_size_button, button_color=button_color, font=font, disabled=disabled, bind_return_key=bind_return_key, focus=focus, pad=pad, key=key)

def Exit(button_text='Exit', size=(None, None), auto_size_button=None, button_color=None, disabled=False, tooltip=None, font=None, bind_return_key=False, focus=False, pad=None, key=None):
    if False:
        for i in range(10):
            print('nop')
    return Button(button_text=button_text, button_type=BUTTON_TYPE_READ_FORM, tooltip=tooltip, size=size, auto_size_button=auto_size_button, button_color=button_color, font=font, disabled=disabled, bind_return_key=bind_return_key, focus=focus, pad=pad, key=key)

def Up(button_text='▲', size=(None, None), auto_size_button=None, button_color=None, disabled=False, tooltip=None, font=None, bind_return_key=True, focus=False, pad=None, key=None):
    if False:
        i = 10
        return i + 15
    return Button(button_text=button_text, button_type=BUTTON_TYPE_READ_FORM, tooltip=tooltip, size=size, auto_size_button=auto_size_button, button_color=button_color, font=font, disabled=disabled, bind_return_key=bind_return_key, focus=focus, pad=pad, key=key)

def Down(button_text='▼', size=(None, None), auto_size_button=None, button_color=None, disabled=False, tooltip=None, font=None, bind_return_key=True, focus=False, pad=None, key=None):
    if False:
        while True:
            i = 10
    return Button(button_text=button_text, button_type=BUTTON_TYPE_READ_FORM, tooltip=tooltip, size=size, auto_size_button=auto_size_button, button_color=button_color, font=font, disabled=disabled, bind_return_key=bind_return_key, focus=focus, pad=pad, key=key)

def Left(button_text='◄', size=(None, None), auto_size_button=None, button_color=None, disabled=False, tooltip=None, font=None, bind_return_key=True, focus=False, pad=None, key=None):
    if False:
        for i in range(10):
            print('nop')
    return Button(button_text=button_text, button_type=BUTTON_TYPE_READ_FORM, tooltip=tooltip, size=size, auto_size_button=auto_size_button, button_color=button_color, font=font, disabled=disabled, bind_return_key=bind_return_key, focus=focus, pad=pad, key=key)

def Right(button_text='►', size=(None, None), auto_size_button=None, button_color=None, disabled=False, tooltip=None, font=None, bind_return_key=True, focus=False, pad=None, key=None):
    if False:
        print('Hello World!')
    return Button(button_text=button_text, button_type=BUTTON_TYPE_READ_FORM, tooltip=tooltip, size=size, auto_size_button=auto_size_button, button_color=button_color, font=font, disabled=disabled, bind_return_key=bind_return_key, focus=focus, pad=pad, key=key)

def Yes(button_text='Yes', size=(None, None), auto_size_button=None, button_color=None, disabled=False, tooltip=None, font=None, bind_return_key=True, focus=False, pad=None, key=None):
    if False:
        while True:
            i = 10
    return Button(button_text=button_text, button_type=BUTTON_TYPE_READ_FORM, tooltip=tooltip, size=size, auto_size_button=auto_size_button, button_color=button_color, font=font, disabled=disabled, bind_return_key=bind_return_key, focus=focus, pad=pad, key=key)

def No(button_text='No', size=(None, None), auto_size_button=None, button_color=None, disabled=False, tooltip=None, font=None, bind_return_key=False, focus=False, pad=None, key=None):
    if False:
        return 10
    return Button(button_text=button_text, button_type=BUTTON_TYPE_READ_FORM, tooltip=tooltip, size=size, auto_size_button=auto_size_button, button_color=button_color, font=font, disabled=disabled, bind_return_key=bind_return_key, focus=focus, pad=pad, key=key)

def Help(button_text='Help', size=(None, None), auto_size_button=None, button_color=None, disabled=False, font=None, tooltip=None, bind_return_key=False, focus=False, pad=None, key=None):
    if False:
        i = 10
        return i + 15
    return Button(button_text=button_text, button_type=BUTTON_TYPE_READ_FORM, tooltip=tooltip, size=size, auto_size_button=auto_size_button, button_color=button_color, font=font, disabled=disabled, bind_return_key=bind_return_key, focus=focus, pad=pad, key=key)

def SimpleButton(button_text, image_filename=None, image_data=None, image_size=(None, None), image_subsample=None, border_width=None, tooltip=None, size=(None, None), auto_size_button=None, button_color=None, font=None, bind_return_key=False, disabled=False, focus=False, pad=None, key=None):
    if False:
        return 10
    return Button(button_text=button_text, button_type=BUTTON_TYPE_CLOSES_WIN, image_filename=image_filename, image_data=image_data, image_size=image_size, image_subsample=image_subsample, border_width=border_width, tooltip=tooltip, disabled=disabled, size=size, auto_size_button=auto_size_button, button_color=button_color, font=font, bind_return_key=bind_return_key, focus=focus, pad=pad, key=key)

def CloseButton(button_text, image_filename=None, image_data=None, image_size=(None, None), image_subsample=None, border_width=None, tooltip=None, size=(None, None), auto_size_button=None, button_color=None, font=None, bind_return_key=False, disabled=False, focus=False, pad=None, key=None):
    if False:
        print('Hello World!')
    return Button(button_text=button_text, button_type=BUTTON_TYPE_CLOSES_WIN, image_filename=image_filename, image_data=image_data, image_size=image_size, image_subsample=image_subsample, border_width=border_width, tooltip=tooltip, disabled=disabled, size=size, auto_size_button=auto_size_button, button_color=button_color, font=font, bind_return_key=bind_return_key, focus=focus, pad=pad, key=key)
CButton = CloseButton

def ReadButton(button_text, image_filename=None, image_data=None, image_size=(None, None), image_subsample=None, border_width=None, tooltip=None, size=(None, None), auto_size_button=None, button_color=None, font=None, bind_return_key=False, disabled=False, focus=False, pad=None, key=None):
    if False:
        return 10
    return Button(button_text=button_text, button_type=BUTTON_TYPE_READ_FORM, image_filename=image_filename, image_data=image_data, image_size=image_size, image_subsample=image_subsample, border_width=border_width, tooltip=tooltip, size=size, disabled=disabled, auto_size_button=auto_size_button, button_color=button_color, font=font, bind_return_key=bind_return_key, focus=focus, pad=pad, key=key)
ReadFormButton = ReadButton
RButton = ReadFormButton

def RealtimeButton(button_text, image_filename=None, image_data=None, image_size=(None, None), image_subsample=None, border_width=None, tooltip=None, size=(None, None), auto_size_button=None, button_color=None, font=None, disabled=False, bind_return_key=False, focus=False, pad=None, key=None):
    if False:
        for i in range(10):
            print('nop')
    return Button(button_text=button_text, button_type=BUTTON_TYPE_REALTIME, image_filename=image_filename, image_data=image_data, image_size=image_size, image_subsample=image_subsample, border_width=border_width, tooltip=tooltip, disabled=disabled, size=size, auto_size_button=auto_size_button, button_color=button_color, font=font, bind_return_key=bind_return_key, focus=focus, pad=pad, key=key)

def DummyButton(button_text, image_filename=None, image_data=None, image_size=(None, None), image_subsample=None, border_width=None, tooltip=None, size=(None, None), auto_size_button=None, button_color=None, font=None, disabled=False, bind_return_key=False, focus=False, pad=None, key=None):
    if False:
        print('Hello World!')
    return Button(button_text=button_text, button_type=BUTTON_TYPE_CLOSES_WIN_ONLY, image_filename=image_filename, image_data=image_data, image_size=image_size, image_subsample=image_subsample, border_width=border_width, tooltip=tooltip, size=size, auto_size_button=auto_size_button, button_color=button_color, font=font, disabled=disabled, bind_return_key=bind_return_key, focus=focus, pad=pad, key=key)

def CalendarButton(button_text, target=(None, None), close_when_date_chosen=True, default_date_m_d_y=(None, None, None), image_filename=None, image_data=None, image_size=(None, None), image_subsample=None, tooltip=None, border_width=None, size=(None, None), auto_size_button=None, button_color=None, disabled=False, font=None, bind_return_key=False, focus=False, pad=None, key=None):
    if False:
        i = 10
        return i + 15
    button = Button(button_text=button_text, button_type=BUTTON_TYPE_CALENDAR_CHOOSER, target=target, image_filename=image_filename, image_data=image_data, image_size=image_size, image_subsample=image_subsample, border_width=border_width, tooltip=tooltip, size=size, auto_size_button=auto_size_button, button_color=button_color, font=font, disabled=disabled, bind_return_key=bind_return_key, focus=focus, pad=pad, key=key)
    button.CalendarCloseWhenChosen = close_when_date_chosen
    button.DefaultDate_M_D_Y = default_date_m_d_y
    return button

def ColorChooserButton(button_text, target=(None, None), image_filename=None, image_data=None, image_size=(None, None), image_subsample=None, tooltip=None, border_width=None, size=(None, None), auto_size_button=None, button_color=None, disabled=False, font=None, bind_return_key=False, focus=False, pad=None, key=None):
    if False:
        return 10
    return Button(button_text=button_text, button_type=BUTTON_TYPE_COLOR_CHOOSER, target=target, image_filename=image_filename, image_data=image_data, image_size=image_size, image_subsample=image_subsample, border_width=border_width, tooltip=tooltip, size=size, auto_size_button=auto_size_button, button_color=button_color, font=font, disabled=disabled, bind_return_key=bind_return_key, focus=focus, pad=pad, key=key)

def AddToReturnDictionary(form, element, value):
    if False:
        print('Hello World!')
    form.ReturnValuesDictionary[element.Key] = value
    return
    if element.Key is None:
        form.ReturnValuesDictionary[form.DictionaryKeyCounter] = value
        element.Key = form.DictionaryKeyCounter
        form.DictionaryKeyCounter += 1
    else:
        form.ReturnValuesDictionary[element.Key] = value

def AddToReturnList(form, value):
    if False:
        while True:
            i = 10
    form.ReturnValuesList.append(value)

def InitializeResults(form):
    if False:
        print('Hello World!')
    BuildResults(form, True, form)
    return

def DecodeRadioRowCol(RadValue):
    if False:
        return 10
    row = RadValue // 1000
    col = RadValue % 1000
    return (row, col)

def EncodeRadioRowCol(row, col):
    if False:
        while True:
            i = 10
    RadValue = row * 1000 + col
    return RadValue

def BuildResults(form, initialize_only, top_level_form):
    if False:
        for i in range(10):
            print('nop')
    form.DictionaryKeyCounter = 0
    form.ReturnValuesDictionary = {}
    form.ReturnValuesList = []
    BuildResultsForSubform(form, initialize_only, top_level_form)
    if not top_level_form.LastButtonClickedWasRealtime:
        top_level_form.LastButtonClicked = None
    return form.ReturnValues

def BuildResultsForSubform(form, initialize_only, top_level_form):
    if False:
        i = 10
        return i + 15
    button_pressed_text = top_level_form.LastButtonClicked
    for (row_num, row) in enumerate(form.Rows):
        for (col_num, element) in enumerate(row):
            if element.Key is not None and WRITE_ONLY_KEY in str(element.Key):
                continue
            value = None
            if element.Type == ELEM_TYPE_COLUMN:
                element.DictionaryKeyCounter = top_level_form.DictionaryKeyCounter
                element.ReturnValuesList = []
                element.ReturnValuesDictionary = {}
                BuildResultsForSubform(element, initialize_only, top_level_form)
                for item in element.ReturnValuesList:
                    AddToReturnList(top_level_form, item)
                if element.UseDictionary:
                    top_level_form.UseDictionary = True
                if element.ReturnValues[0] is not None:
                    button_pressed_text = element.ReturnValues[0]
            if element.Type == ELEM_TYPE_FRAME:
                element.DictionaryKeyCounter = top_level_form.DictionaryKeyCounter
                element.ReturnValuesList = []
                element.ReturnValuesDictionary = {}
                BuildResultsForSubform(element, initialize_only, top_level_form)
                for item in element.ReturnValuesList:
                    AddToReturnList(top_level_form, item)
                if element.UseDictionary:
                    top_level_form.UseDictionary = True
                if element.ReturnValues[0] is not None:
                    button_pressed_text = element.ReturnValues[0]
            if element.Type == ELEM_TYPE_TAB_GROUP:
                element.DictionaryKeyCounter = top_level_form.DictionaryKeyCounter
                element.ReturnValuesList = []
                element.ReturnValuesDictionary = {}
                BuildResultsForSubform(element, initialize_only, top_level_form)
                for item in element.ReturnValuesList:
                    AddToReturnList(top_level_form, item)
                if element.UseDictionary:
                    top_level_form.UseDictionary = True
                if element.ReturnValues[0] is not None:
                    button_pressed_text = element.ReturnValues[0]
            if element.Type == ELEM_TYPE_TAB:
                element.DictionaryKeyCounter = top_level_form.DictionaryKeyCounter
                element.ReturnValuesList = []
                element.ReturnValuesDictionary = {}
                BuildResultsForSubform(element, initialize_only, top_level_form)
                for item in element.ReturnValuesList:
                    AddToReturnList(top_level_form, item)
                if element.UseDictionary:
                    top_level_form.UseDictionary = True
                if element.ReturnValues[0] is not None:
                    button_pressed_text = element.ReturnValues[0]
            if not initialize_only:
                if element.Type == ELEM_TYPE_INPUT_TEXT:
                    element = element
                    value = element.Widget.get_value()
                    if not top_level_form.NonBlocking and (not element.do_not_clear) and (not top_level_form.ReturnKeyboardEvents):
                        element.Widget.set_value('')
                elif element.Type == ELEM_TYPE_INPUT_CHECKBOX:
                    element = element
                    value = element.Widget.get_value()
                elif element.Type == ELEM_TYPE_INPUT_RADIO:
                    value = False
                elif element.Type == ELEM_TYPE_BUTTON:
                    if top_level_form.LastButtonClicked == element.ButtonText:
                        button_pressed_text = top_level_form.LastButtonClicked
                        if element.BType != BUTTON_TYPE_REALTIME:
                            top_level_form.LastButtonClicked = None
                    if element.BType == BUTTON_TYPE_CALENDAR_CHOOSER:
                        try:
                            value = element.TKCal.selection
                        except:
                            value = None
                    else:
                        try:
                            value = element.TKStringVar.get()
                        except:
                            value = None
                elif element.Type == ELEM_TYPE_INPUT_COMBO:
                    element = element
                    value = element.Widget.get_value()
                elif element.Type == ELEM_TYPE_INPUT_OPTION_MENU:
                    value = None
                elif element.Type == ELEM_TYPE_INPUT_LISTBOX:
                    element = element
                    value = element.Widget.get_value()
                    value = [value]
                elif element.Type == ELEM_TYPE_INPUT_SPIN:
                    element = element
                    value = element.Widget.get_value()
                elif element.Type == ELEM_TYPE_INPUT_SLIDER:
                    element = element
                    value = element.Widget.get_value()
                elif element.Type == ELEM_TYPE_INPUT_MULTILINE:
                    element = element
                    if element.WriteOnly:
                        continue
                    value = element.Widget.get_value()
                elif element.Type == ELEM_TYPE_TAB_GROUP:
                    try:
                        value = element.TKNotebook.tab(element.TKNotebook.index('current'))['text']
                        tab_key = element.FindKeyFromTabName(value)
                        if tab_key is not None:
                            value = tab_key
                    except:
                        value = None
                elif element.Type == ELEM_TYPE_TABLE:
                    element = element
                    value = [element.SelectedRow]
                elif element.Type == ELEM_TYPE_TREE:
                    value = element.SelectedRows
                elif element.Type == ELEM_TYPE_GRAPH:
                    value = element.ClickPosition
                elif element.Type == ELEM_TYPE_MENUBAR:
                    value = element.MenuItemChosen
            else:
                value = None
            if element.Type != ELEM_TYPE_BUTTON and element.Type != ELEM_TYPE_TEXT and (element.Type != ELEM_TYPE_IMAGE) and (element.Type != ELEM_TYPE_OUTPUT) and (element.Type != ELEM_TYPE_PROGRESS_BAR) and (element.Type != ELEM_TYPE_COLUMN) and (element.Type != ELEM_TYPE_FRAME) and (element.Type != ELEM_TYPE_TAB):
                AddToReturnList(form, value)
                AddToReturnDictionary(top_level_form, element, value)
            elif element.Type == ELEM_TYPE_BUTTON and element.BType == BUTTON_TYPE_CALENDAR_CHOOSER and (element.Target == (None, None)) or (element.Type == ELEM_TYPE_BUTTON and element.BType == BUTTON_TYPE_COLOR_CHOOSER and (element.Target == (None, None))) or (element.Type == ELEM_TYPE_BUTTON and element.Key is not None and (element.BType in (BUTTON_TYPE_SAVEAS_FILE, BUTTON_TYPE_BROWSE_FILE, BUTTON_TYPE_BROWSE_FILES, BUTTON_TYPE_BROWSE_FOLDER))):
                AddToReturnList(form, value)
                AddToReturnDictionary(top_level_form, element, value)
    try:
        if form.ReturnKeyboardEvents and form.LastKeyboardEvent is not None:
            button_pressed_text = form.LastKeyboardEvent
            form.LastKeyboardEvent = None
    except:
        pass
    try:
        form.ReturnValuesDictionary.pop(None, None)
    except:
        pass
    if not form.UseDictionary:
        form.ReturnValues = (button_pressed_text, form.ReturnValuesList)
    else:
        form.ReturnValues = (button_pressed_text, form.ReturnValuesDictionary)
    return form.ReturnValues

def _FillFormWithValues(form, values_dict):
    if False:
        print('Hello World!')
    _FillSubformWithValues(form, values_dict)

def _FillSubformWithValues(form, values_dict):
    if False:
        for i in range(10):
            print('nop')
    for (row_num, row) in enumerate(form.Rows):
        for (col_num, element) in enumerate(row):
            value = None
            if element.Type == ELEM_TYPE_COLUMN:
                _FillSubformWithValues(element, values_dict)
            if element.Type == ELEM_TYPE_FRAME:
                _FillSubformWithValues(element, values_dict)
            if element.Type == ELEM_TYPE_TAB_GROUP:
                _FillSubformWithValues(element, values_dict)
            if element.Type == ELEM_TYPE_TAB:
                _FillSubformWithValues(element, values_dict)
            try:
                value = values_dict[element.Key]
            except:
                continue
            if element.Type == ELEM_TYPE_INPUT_TEXT:
                element.Update(value)
            elif element.Type == ELEM_TYPE_INPUT_CHECKBOX:
                element.Update(value)
            elif element.Type == ELEM_TYPE_INPUT_RADIO:
                element.Update(value)
            elif element.Type == ELEM_TYPE_INPUT_COMBO:
                element.Update(value)
            elif element.Type == ELEM_TYPE_INPUT_OPTION_MENU:
                element.Update(value)
            elif element.Type == ELEM_TYPE_INPUT_LISTBOX:
                element.SetValue(value)
            elif element.Type == ELEM_TYPE_INPUT_SLIDER:
                element.Update(value)
            elif element.Type == ELEM_TYPE_INPUT_MULTILINE:
                element.Update(value)
            elif element.Type == ELEM_TYPE_INPUT_SPIN:
                element.Update(value)
            elif element.Type == ELEM_TYPE_BUTTON:
                element.Update(value)

def _FindElementFromKeyInSubForm(form, key):
    if False:
        print('Hello World!')
    for (row_num, row) in enumerate(form.Rows):
        for (col_num, element) in enumerate(row):
            if element.Type == ELEM_TYPE_COLUMN:
                matching_elem = _FindElementFromKeyInSubForm(element, key)
                if matching_elem is not None:
                    return matching_elem
            if element.Type == ELEM_TYPE_FRAME:
                matching_elem = _FindElementFromKeyInSubForm(element, key)
                if matching_elem is not None:
                    return matching_elem
            if element.Type == ELEM_TYPE_TAB_GROUP:
                matching_elem = _FindElementFromKeyInSubForm(element, key)
                if matching_elem is not None:
                    return matching_elem
            if element.Type == ELEM_TYPE_TAB:
                matching_elem = _FindElementFromKeyInSubForm(element, key)
                if matching_elem is not None:
                    return matching_elem
            if element.Key == key:
                return element

def _FindElementWithFocusInSubForm(form):
    if False:
        for i in range(10):
            print('nop')
    for (row_num, row) in enumerate(form.Rows):
        for (col_num, element) in enumerate(row):
            if element.Type == ELEM_TYPE_COLUMN:
                matching_elem = _FindElementWithFocusInSubForm(element)
                if matching_elem is not None:
                    return matching_elem
            if element.Type == ELEM_TYPE_FRAME:
                matching_elem = _FindElementWithFocusInSubForm(element)
                if matching_elem is not None:
                    return matching_elem
            if element.Type == ELEM_TYPE_TAB_GROUP:
                matching_elem = _FindElementWithFocusInSubForm(element)
                if matching_elem is not None:
                    return matching_elem
            if element.Type == ELEM_TYPE_TAB:
                matching_elem = _FindElementWithFocusInSubForm(element)
                if matching_elem is not None:
                    return matching_elem
            if element.Type == ELEM_TYPE_INPUT_TEXT:
                if element.TKEntry is not None:
                    if element.TKEntry is element.TKEntry.focus_get():
                        return element
            if element.Type == ELEM_TYPE_INPUT_MULTILINE:
                if element.TKText is not None:
                    if element.TKText is element.TKText.focus_get():
                        return element

def AddMenuItem(top_menu, sub_menu_info, element, is_sub_menu=False, skip=False):
    if False:
        return 10
    return_val = None
    if type(sub_menu_info) is str:
        if not is_sub_menu and (not skip):
            pos = sub_menu_info.find('&')
            if pos != -1:
                if pos == 0 or sub_menu_info[pos - 1] != '\\':
                    sub_menu_info = sub_menu_info[:pos] + sub_menu_info[pos + 1:]
            if sub_menu_info == '---':
                pass
            else:
                try:
                    item_without_key = sub_menu_info[:sub_menu_info.index(MENU_KEY_SEPARATOR)]
                except:
                    item_without_key = sub_menu_info
                if item_without_key[0] == MENU_DISABLED_CHARACTER:
                    menu_item = remi.gui.MenuItem(item_without_key[1:], width=100, height=30)
                    menu_item.set_enabled(False)
                    top_menu.append([menu_item])
                else:
                    menu_item = remi.gui.MenuItem(item_without_key, width=100, height=30)
                    top_menu.append([menu_item])
                menu_item.onclick.connect(element._ChangedCallbackMenu, sub_menu_info)
    else:
        i = 0
        while i < len(sub_menu_info):
            item = sub_menu_info[i]
            if i != len(sub_menu_info) - 1:
                if type(sub_menu_info[i + 1]) == list:
                    pos = sub_menu_info[i].find('&')
                    if pos != -1:
                        if pos == 0 or sub_menu_info[i][pos - 1] != '\\':
                            sub_menu_info[i] = sub_menu_info[i][:pos] + sub_menu_info[i][pos + 1:]
                    if sub_menu_info[i][0] == MENU_DISABLED_CHARACTER:
                        new_menu = remi.gui.MenuItem(sub_menu_info[i][len(MENU_DISABLED_CHARACTER):], width=100, height=30)
                        new_menu.set_enabled(False)
                    else:
                        new_menu = remi.gui.MenuItem(sub_menu_info[i], width=100, height=30)
                    top_menu.append([new_menu])
                    return_val = new_menu
                    AddMenuItem(new_menu, sub_menu_info[i + 1], element, is_sub_menu=True)
                    i += 1
                else:
                    AddMenuItem(top_menu, item, element)
            else:
                AddMenuItem(top_menu, item, element)
            i += 1
    return return_val
'\n          :::::::::       ::::::::::         :::   :::       ::::::::::: \n         :+:    :+:      :+:               :+:+: :+:+:          :+:      \n        +:+    +:+      +:+              +:+ +:+:+ +:+         +:+       \n       +#++:++#:       +#++:++#         +#+  +:+  +#+         +#+        \n      +#+    +#+      +#+              +#+       +#+         +#+         \n     #+#    #+#      #+#              #+#       #+#         #+#          \n    ###    ###      ##########       ###       ###     ###########    \n'

def PackFormIntoFrame(form, containing_frame, toplevel_form):
    if False:
        while True:
            i = 10

    def CharWidthInPixels():
        if False:
            while True:
                i = 10
        return tkinter.font.Font().measure('A')

    def pad_widget(widget):
        if False:
            print('Hello World!')
        lrsizer = wx.BoxSizer(wx.HORIZONTAL)
        if full_element_pad[1] == full_element_pad[3]:
            lrsizer.Add(widget, 0, wx.LEFT | wx.RIGHT, border=full_element_pad[1])
        else:
            sizer = wx.BoxSizer(wx.HORIZONTAL)
            sizer.Add(widget, 0, wx.LEFT, border=full_element_pad[3])
            lrsizer.Add(sizer, 0, wx.RIGHT, border=full_element_pad[1])
        top_bottom_sizer = wx.BoxSizer(wx.HORIZONTAL)
        if full_element_pad[0] == full_element_pad[2]:
            top_bottom_sizer.Add(lrsizer, 0, wx.TOP | wx.BOTTOM, border=full_element_pad[0])
        else:
            sizer = wx.BoxSizer(wx.HORIZONTAL)
            sizer.Add(lrsizer, 0, wx.TOP, border=full_element_pad[0])
            top_bottom_sizer.Add(sizer, 0, wx.BOTTOM, border=full_element_pad[2])
        return top_bottom_sizer

    def do_font_and_color(widget):
        if False:
            return 10
        font_info = font_parse_string(font)
        widget.style['font-family'] = font_info[0]
        if element.BackgroundColor not in (None, COLOR_SYSTEM_DEFAULT):
            widget.style['background-color'] = element.BackgroundColor
        if element.TextColor not in (None, COLOR_SYSTEM_DEFAULT):
            widget.style['color'] = element.TextColor
        widget.style['font-size'] = '{}px'.format(font_info[1])
        if element_size[0]:
            size = convert_tkinter_size_to_Wx(element_size)
            widget.style['height'] = '{}px'.format(size[1])
            widget.style['width'] = '{}px'.format(size[0])
        widget.style['margin'] = '{}px {}px {}px {}px'.format(*full_element_pad)
        if element.Disabled:
            widget.set_enabled(False)
        if not element.Visible:
            widget.attributes['hidden'] = 'true'
        if element.Tooltip is not None:
            widget.attributes['title'] = element.Tooltip
    border_depth = toplevel_form.BorderDepth if toplevel_form.BorderDepth is not None else DEFAULT_BORDER_WIDTH
    focus_set = False
    for (row_num, flex_row) in enumerate(form.Rows):
        tk_row_frame = remi.gui.HBox()
        tk_row_frame.style['align-items'] = 'flex-start'
        if form.ElementJustification.startswith('c'):
            tk_row_frame.style['margin-left'] = 'auto'
            tk_row_frame.style['margin-right'] = 'auto'
        elif form.ElementJustification.startswith('r'):
            tk_row_frame.style['margin-left'] = 'auto'
        else:
            tk_row_frame.style['margin-right'] = 'auto'
        if form.BackgroundColor not in (None, COLOR_SYSTEM_DEFAULT):
            tk_row_frame.style['background-color'] = form.BackgroundColor
        for (col_num, element) in enumerate(flex_row):
            element.ParentForm = toplevel_form
            if toplevel_form.Font and (element.Font == DEFAULT_FONT or not element.Font):
                font = toplevel_form.Font
            elif element.Font is not None:
                font = element.Font
            else:
                font = DEFAULT_FONT
            if element.AutoSizeText is not None:
                auto_size_text = element.AutoSizeText
            elif toplevel_form.AutoSizeText is not None:
                auto_size_text = toplevel_form.AutoSizeText
            else:
                auto_size_text = DEFAULT_AUTOSIZE_TEXT
            element_type = element.Type
            text_color = element.TextColor
            element_size = element.Size
            if element_size == (None, None) and element_type != ELEM_TYPE_BUTTON:
                element_size = toplevel_form.DefaultElementSize
            elif element_size == (None, None) and element_type == ELEM_TYPE_BUTTON:
                element_size = toplevel_form.DefaultButtonElementSize
            else:
                auto_size_text = False
            full_element_pad = [0, 0, 0, 0]
            elementpad = element.Pad if element.Pad is not None else toplevel_form.ElementPadding
            if type(elementpad[0]) != tuple:
                full_element_pad[1] = full_element_pad[3] = elementpad[0]
            else:
                (full_element_pad[3], full_element_pad[1]) = elementpad[0]
            if type(elementpad[1]) != tuple:
                full_element_pad[0] = full_element_pad[2] = elementpad[1]
            else:
                (full_element_pad[0], full_element_pad[2]) = elementpad[1]
            if element_type == ELEM_TYPE_COLUMN:
                element = element
                element.Widget = column_widget = remi.gui.VBox()
                if element.BackgroundColor not in (None, COLOR_SYSTEM_DEFAULT):
                    column_widget.style['background-color'] = element.BackgroundColor
                PackFormIntoFrame(element, column_widget, toplevel_form)
                tk_row_frame.append(element.Widget)
            elif element_type == ELEM_TYPE_TEXT:
                element = element
                element.Widget = remi.gui.Label(element.DisplayText)
                do_font_and_color(element.Widget)
                if auto_size_text and element.Size == (None, None):
                    del element.Widget.style['width']
                if element.Justification:
                    if element.Justification.startswith('c'):
                        element.Widget.style['text-align'] = 'center'
                    elif element.Justification.startswith('r'):
                        element.Widget.style['text-align'] = 'right'
                if element.ClickSubmits:
                    element.Widget.onclick.connect(element._ChangedCallback)
                tk_row_frame.append(element.Widget)
            elif element_type == ELEM_TYPE_BUTTON:
                element = element
                size = convert_tkinter_size_to_Wx(element_size)
                element.Widget = remi.gui.Button(element.ButtonText, width=size[0], height=size[1], margin='10px')
                element.Widget.onclick.connect(element._ButtonCallBack)
                do_font_and_color(element.Widget)
                if element.AutoSizeButton or ((toplevel_form.AutoSizeButtons and element.AutoSizeButton is not False) and element.Size == (None, None)):
                    del element.Widget.style['width']
                if element.ImageFilename:
                    element.ImageWidget = SuperImage(element.ImageFilename if element.ImageFilename is not None else element.ImageData)
                    element.Widget.append(element.ImageWidget)
                tk_row_frame.append(element.Widget)
            elif element_type == ELEM_TYPE_INPUT_TEXT:
                element = element
                element.Widget = InputText.TextInput_raw_onkeyup(hint=element.DefaultText)
                do_font_and_color(element.Widget)
                if element.ChangeSubmits:
                    element.Widget.onkeyup.connect(element._InputTextCallback)
                tk_row_frame.append(element.Widget)
            elif element_type == ELEM_TYPE_INPUT_COMBO:
                element = element
                element.Widget = remi.gui.DropDown.new_from_list(element.Values)
                if element.DefaultValue is not None:
                    element.Widget.select_by_value(element.DefaultValue)
                do_font_and_color(element.Widget)
                if element.ChangeSubmits:
                    element.Widget.onchange.connect(element._ChangedCallback)
                tk_row_frame.append(element.Widget)
            elif element_type == ELEM_TYPE_INPUT_OPTION_MENU:
                element.Widget = remi.gui.FileUploader('./', width=200, height=30, margin='10px')
                tk_row_frame.append(element.Widget)
                pass
            elif element_type == ELEM_TYPE_INPUT_LISTBOX:
                element = element
                element.Widget = remi.gui.ListView.new_from_list(element.Values)
                do_font_and_color(element.Widget)
                if element.ChangeSubmits:
                    element.Widget.onselection.connect(element._ChangedCallback)
                tk_row_frame.append(element.Widget)
            elif element_type == ELEM_TYPE_INPUT_MULTILINE:
                element = element
                element.Widget = remi.gui.TextInput(single_line=False, hint=element.DefaultText)
                do_font_and_color(element.Widget)
                if element.ChangeSubmits:
                    element.Widget.onkeydown.connect(element._InputTextCallback)
                tk_row_frame.append(element.Widget)
            elif element_type == ELEM_TYPE_INPUT_CHECKBOX:
                element = element
                element.Widget = remi.gui.CheckBoxLabel(element.Text)
                if element.InitialState:
                    element.Widget.set_value(element.InitialState)
                if element.ChangeSubmits:
                    element.Widget.onchange.connect(element._ChangedCallback)
                do_font_and_color(element.Widget)
                tk_row_frame.append(element.Widget)
            elif element_type == ELEM_TYPE_PROGRESS_BAR:
                pass
            elif element_type == ELEM_TYPE_INPUT_RADIO:
                pass
            elif element_type == ELEM_TYPE_INPUT_SPIN:
                element = element
                element.Widget = remi.gui.SpinBox(50, 0, 100)
                if element.DefaultValue is not None:
                    element.Widget.set_value(element.DefaultValue)
                do_font_and_color(element.Widget)
                if element.ChangeSubmits:
                    element.Widget.onchange.connect(element._ChangedCallback)
                tk_row_frame.append(element.Widget)
            elif element_type == ELEM_TYPE_OUTPUT:
                element = element
                element.Widget = remi.gui.TextInput(single_line=False)
                element.Disabled = True
                do_font_and_color(element.Widget)
                tk_row_frame.append(element.Widget)
                toplevel_form.OutputElementForStdOut = element
                Window.stdout_is_rerouted = True
                Window.stdout_string_io = StringIO()
                sys.stdout = Window.stdout_string_io
            elif element_type == ELEM_TYPE_MULTILINE_OUTPUT:
                element = element
                element.Widget = remi.gui.TextInput(single_line=False)
                element.Disabled = True
                do_font_and_color(element.Widget)
                tk_row_frame.append(element.Widget)
                if element.DefaultText:
                    element.Widget.set_value(element.DefaultText)
            elif element_type == ELEM_TYPE_IMAGE:
                element = element
                element.Widget = SuperImage(element.Filename if element.Filename is not None else element.Data)
                if element.Filename is not None:
                    element.Widget.load(element.Filename)
                do_font_and_color(element.Widget)
                if element.EnableEvents:
                    element.Widget.onclick.connect(element._ChangedCallback)
                tk_row_frame.append(element.Widget)
            elif element_type == ELEM_TYPE_CANVAS:
                pass
            elif element_type == ELEM_TYPE_GRAPH:
                element = element
                element.Widget = remi.gui.Svg(width=element.CanvasSize[0], height=element.CanvasSize[1])
                element.SvgGroup = remi.gui.SvgSubcontainer(0, 0, '100%', '100%')
                element.Widget.append([element.SvgGroup])
                do_font_and_color(element.Widget)
                if element.ChangeSubmits:
                    element.Widget.onmouseup.connect(element._MouseUpCallback)
                if element.DragSubmits:
                    element.Widget.onmousedown.connect(element._MouseDownCallback)
                    element.Widget.onmouseup.connect(element._MouseUpCallback)
                    element.Widget.onmousemove.connect(element._DragCallback)
                tk_row_frame.append(element.Widget)
            elif element_type == ELEM_TYPE_MENUBAR:
                element = element
                menu = remi.gui.Menu(width='100%', height=str(element_size[1]))
                element_size = (0, 0)
                do_font_and_color(menu)
                menu_def = element.MenuDefinition
                for menu_entry in menu_def:
                    pos = menu_entry[0].find('&')
                    if pos != -1:
                        if pos == 0 or menu_entry[0][pos - 1] != '\\':
                            menu_entry[0] = menu_entry[0][:pos] + menu_entry[0][pos + 1:]
                    if menu_entry[0][0] == MENU_DISABLED_CHARACTER:
                        item = remi.gui.MenuItem(menu_entry[0][1:], width=100, height=element_size[1])
                        item.set_enabled(False)
                    else:
                        item = remi.gui.MenuItem(menu_entry[0], width=100, height=element_size[1])
                    do_font_and_color(item)
                    menu.append([item])
                    if len(menu_entry) > 1:
                        AddMenuItem(item, menu_entry[1], element)
                element.Widget = menubar = remi.gui.MenuBar(width='100%', height='30px')
                element.Widget.style['z-index'] = '1'
                menubar.append(menu)
                containing_frame.append(element.Widget)
            elif element_type == ELEM_TYPE_FRAME:
                element = element
                element.Widget = column_widget = CLASSframe(element.Title)
                if element.BackgroundColor not in (None, COLOR_SYSTEM_DEFAULT):
                    column_widget.style['background-color'] = element.BackgroundColor
                PackFormIntoFrame(element, column_widget, toplevel_form)
                tk_row_frame.append(element.Widget)
            elif element_type == ELEM_TYPE_TAB:
                element = element
                element.Widget = remi.gui.VBox()
                if element.Justification.startswith('c'):
                    element.Widget.style['align-items'] = 'center'
                    element.Widget.style['justify-content'] = 'center'
                else:
                    element.Widget.style['justify-content'] = 'flex-start'
                    element.Widget.style['align-items'] = 'baseline'
                if element.BackgroundColor not in (None, COLOR_SYSTEM_DEFAULT):
                    element.Widget.style['background-color'] = element.BackgroundColor
                if element.BackgroundColor not in (None, COLOR_SYSTEM_DEFAULT):
                    element.Widget.style['background-color'] = element.BackgroundColor
                PackFormIntoFrame(element, element.Widget, toplevel_form)
                containing_frame.add_tab(element.Widget, element.Title, None)
            elif element_type == ELEM_TYPE_TAB_GROUP:
                element = element
                element.Widget = remi.gui.TabBox()
                PackFormIntoFrame(element, element.Widget, toplevel_form)
                tk_row_frame.append(element.Widget)
            elif element_type == ELEM_TYPE_INPUT_SLIDER:
                element = element
                orient = remi.gui.Container.LAYOUT_HORIZONTAL if element.Orientation.lower().startswith('h') else remi.gui.Container.LAYOUT_VERTICAL
                element.Widget = remi.gui.Slider(layout_orientation=orient, default_value=element.DefaultValue, min=element.Range[0], max=element.Range[1], step=element.Resolution)
                if element.DefaultValue:
                    element.Widget.set_value(element.DefaultValue)
                do_font_and_color(element.Widget)
                if element.ChangeSubmits:
                    element.Widget.onchange.connect(element._SliderCallback)
                element.Widget.style['orientation'] = 'vertical'
                element.Widget.attributes['orientation'] = 'vertical'
                tk_row_frame.append(element.Widget)
            elif element_type == ELEM_TYPE_TABLE:
                element = element
                new_table = []
                for (row_num, row) in enumerate(element.Values):
                    new_row = [str(item) for item in row]
                    if element.DisplayRowNumbers:
                        new_row = [element.RowHeaderText if row_num == 0 else str(row_num + element.StartingRowNumber)] + new_row
                    new_table.append(new_row)
                element.Widget = remi.gui.Table.new_from_list(new_table)
                do_font_and_color(element.Widget)
                tk_row_frame.append(element.Widget)
                element.Widget.on_table_row_click.connect(element._on_table_row_click)
                pass
            elif element_type == ELEM_TYPE_TREE:
                pass
            elif element_type == ELEM_TYPE_SEPARATOR:
                pass
        if not type(containing_frame) == remi.gui.TabBox:
            containing_frame.append(tk_row_frame)
    return

def setup_remi_window(app: Window.MyApp, window: Window):
    if False:
        while True:
            i = 10
    master_widget = remi.gui.VBox()
    master_widget.style['justify-content'] = 'flex-start'
    master_widget.style['align-items'] = 'baseline'
    if window.BackgroundColor not in (None, COLOR_SYSTEM_DEFAULT):
        master_widget.style['background-color'] = window.BackgroundColor
    try:
        PackFormIntoFrame(window, master_widget, window)
    except:
        print('* ERROR PACKING FORM *')
        print(traceback.format_exc())
    if window.BackgroundImage:
        master_widget.style['background-image'] = "url('{}')".format('/' + window.BackgroundImage)
    if not window.DisableClose:
        tag = remi.gui.Tag(_type='script')
        tag.add_child('javascript', 'window.onunload=function(e){sendCallback(\'%s\',\'%s\');return "close?";};' % (str(id(app)), 'on_window_close'))
        master_widget.add_child('onunloadevent', tag)
    if window.ReturnKeyboardEvents:
        app.page.children['body'].onkeyup.connect(window.on_key_up)
    if window.ReturnKeyDownEvents:
        app.page.children['body'].onkeydown.connect(window.on_key_down)
    return master_widget

def StartupTK(window: Window):
    if False:
        print('Hello World!')
    global _my_windows
    _my_windows.Increment()
    InitializeResults(window)
    if len(Window.active_windows) == 0:
        window.thread_id = threading.Thread(target=window.remi_thread, daemon=True)
        window.thread_id.daemon = True
        window.thread_id.start()
        item = window.MessageQueue.get()
        Window.active_windows.append(window)
        Window.App = window.App
    else:
        master_widget = setup_remi_window(Window.App, window)
        window.master_widget = master_widget
        Window.active_windows.append(window)
        Window.App.set_root_widget(master_widget)
    return

def _GetNumLinesNeeded(text, max_line_width):
    if False:
        while True:
            i = 10
    if max_line_width == 0:
        return 1
    lines = text.split('\n')
    num_lines = len(lines)
    max_line_len = max([len(l) for l in lines])
    lines_used = []
    for L in lines:
        lines_used.append(len(L) // max_line_width + (len(L) % max_line_width > 0))
    total_lines_needed = sum(lines_used)
    return total_lines_needed

def ConvertArgsToSingleString(*args):
    if False:
        i = 10
        return i + 15
    (max_line_total, width_used, total_lines) = (0, 0, 0)
    single_line_message = ''
    for message in args:
        message = str(message)
        longest_line_len = max([len(l) for l in message.split('\n')])
        width_used = max(longest_line_len, width_used)
        max_line_total = max(max_line_total, width_used)
        lines_needed = _GetNumLinesNeeded(message, width_used)
        total_lines += lines_needed
        single_line_message += message + '\n'
    return (single_line_message, width_used, total_lines)

def _ProgressMeter(title, max_value, *args, orientation=None, bar_color=(None, None), button_color=None, size=DEFAULT_PROGRESS_BAR_SIZE, border_width=None, grab_anywhere=False):
    if False:
        return 10
    "\n    Create and show a form on tbe caller's behalf.\n    :param title:\n    :param max_value:\n    :param args: ANY number of arguments the caller wants to display\n    :param orientation:\n    :param bar_color:\n    :param size:\n    :param Style:\n    :param StyleOffset:\n    :return: ProgressBar object that is in the form\n    "
    local_orientation = DEFAULT_METER_ORIENTATION if orientation is None else orientation
    local_border_width = DEFAULT_PROGRESS_BAR_BORDER_WIDTH if border_width is None else border_width
    bar2 = ProgressBar(max_value, orientation=local_orientation, size=size, bar_color=bar_color, border_width=local_border_width, relief=DEFAULT_PROGRESS_BAR_RELIEF)
    form = Window(title, auto_size_text=True, grab_anywhere=grab_anywhere)
    if local_orientation[0].lower() == 'h':
        (single_line_message, width, height) = ConvertArgsToSingleString(*args)
        bar2.TextToDisplay = single_line_message
        bar2.TextToDisplay = single_line_message
        bar2.MaxValue = max_value
        bar2.CurrentValue = 0
        bar_text = Text(single_line_message, size=(width, height + 3), auto_size_text=True)
        form.AddRow(bar_text)
        form.AddRow(bar2)
        form.AddRow(CloseButton('Cancel', button_color=button_color))
    else:
        (single_line_message, width, height) = ConvertArgsToSingleString(*args)
        bar2.TextToDisplay = single_line_message
        bar2.MaxValue = max_value
        bar2.CurrentValue = 0
        bar_text = Text(single_line_message, size=(width, height + 3), auto_size_text=True)
        form.AddRow(bar2, bar_text)
        form.AddRow(CloseButton('Cancel', button_color=button_color))
    form.NonBlocking = True
    form.Show(non_blocking=True)
    return (bar2, bar_text)

def _ProgressMeterUpdate(bar, value, text_elem, *args):
    if False:
        while True:
            i = 10
    '\n    Update the progress meter for a form\n    :param form: class ProgressBar\n    :param value: int\n    :return: True if not cancelled, OK....False if Error\n    '
    global _my_windows
    if bar == None:
        return False
    if bar.BarExpired:
        return False
    (message, w, h) = ConvertArgsToSingleString(*args)
    text_elem.Update(message)
    bar.CurrentValue = value
    rc = bar.UpdateBar(value)
    if value >= bar.MaxValue or not rc:
        bar.BarExpired = True
        bar.ParentForm._Close()
        if rc:
            _my_windows.Decrement()
    if bar.ParentForm.RootNeedsDestroying:
        try:
            bar.ParentForm.TKroot.destroy()
        except:
            pass
        bar.ParentForm.RootNeedsDestroying = False
        return False
    return rc

class EasyProgressMeterDataClass:

    def __init__(self, title='', current_value=1, max_value=10, start_time=None, stat_messages=()):
        if False:
            return 10
        self.Title = title
        self.CurrentValue = current_value
        self.MaxValue = max_value
        self.StartTime = start_time
        self.StatMessages = stat_messages
        self.ParentForm = None
        self.MeterID = None
        self.MeterText = None

    def ComputeProgressStats(self):
        if False:
            return 10
        utc = datetime.datetime.utcnow()
        time_delta = utc - self.StartTime
        total_seconds = time_delta.total_seconds()
        if not total_seconds:
            total_seconds = 1
        try:
            time_per_item = total_seconds / self.CurrentValue
        except:
            time_per_item = 1
        seconds_remaining = (self.MaxValue - self.CurrentValue) * time_per_item
        time_remaining = str(datetime.timedelta(seconds=seconds_remaining))
        time_remaining_short = time_remaining.split('.')[0]
        time_delta_short = str(time_delta).split('.')[0]
        total_time = time_delta + datetime.timedelta(seconds=seconds_remaining)
        total_time_short = str(total_time).split('.')[0]
        self.StatMessages = ['{} of {}'.format(self.CurrentValue, self.MaxValue), '{} %'.format(100 * self.CurrentValue // self.MaxValue), '', ' {:6.2f} Iterations per Second'.format(self.CurrentValue / total_seconds), ' {:6.2f} Seconds per Iteration'.format(total_seconds / (self.CurrentValue if self.CurrentValue else 1)), '', '{} Elapsed Time'.format(time_delta_short), '{} Time Remaining'.format(time_remaining_short), '{} Estimated Total Time'.format(total_time_short)]
        return

def EasyProgressMeter(title, current_value, max_value, *args, orientation=None, bar_color=(None, None), button_color=None, size=DEFAULT_PROGRESS_BAR_SIZE, border_width=None):
    if False:
        i = 10
        return i + 15
    "\n    A ONE-LINE progress meter. Add to your code where ever you need a meter. No need for a second\n    function call before your loop. You've got enough code to write!\n    :param title: Title will be shown on the window\n    :param current_value: Current count of your items\n    :param max_value: Max value your count will ever reach. This indicates it should be closed\n    :param args:  VARIABLE number of arguements... you request it, we'll print it no matter what the item!\n    :param orientation:\n    :param bar_color:\n    :param size:\n    :param Style:\n    :param StyleOffset:\n    :return: False if should stop the meter\n    "
    local_border_width = DEFAULT_PROGRESS_BAR_BORDER_WIDTH if not border_width else border_width
    EasyProgressMeter.Data = getattr(EasyProgressMeter, 'Data', EasyProgressMeterDataClass())
    if EasyProgressMeter.Data.MeterID is None:
        print('Please change your call of EasyProgressMeter to use OneLineProgressMeter. EasyProgressMeter will be removed soon')
        if int(current_value) >= int(max_value):
            return False
        del EasyProgressMeter.Data
        EasyProgressMeter.Data = EasyProgressMeterDataClass(title, 1, int(max_value), datetime.datetime.utcnow(), [])
        EasyProgressMeter.Data.ComputeProgressStats()
        message = '\n'.join([line for line in EasyProgressMeter.Data.StatMessages])
        (EasyProgressMeter.Data.MeterID, EasyProgressMeter.Data.MeterText) = _ProgressMeter(title, int(max_value), message, *args, orientation=orientation, bar_color=bar_color, size=size, button_color=button_color, border_width=local_border_width)
        EasyProgressMeter.Data.ParentForm = EasyProgressMeter.Data.MeterID.ParentForm
        return True
    if EasyProgressMeter.Data.MaxValue == max_value and EasyProgressMeter.Data.CurrentValue == current_value:
        return True
    if EasyProgressMeter.Data.MaxValue != int(max_value):
        EasyProgressMeter.Data.MeterID = None
        EasyProgressMeter.Data.ParentForm = None
        del EasyProgressMeter.Data
        EasyProgressMeter.Data = EasyProgressMeterDataClass()
        return True
    EasyProgressMeter.Data.CurrentValue = int(current_value)
    EasyProgressMeter.Data.MaxValue = int(max_value)
    EasyProgressMeter.Data.ComputeProgressStats()
    message = ''
    for line in EasyProgressMeter.Data.StatMessages:
        message = message + str(line) + '\n'
    message = '\n'.join(EasyProgressMeter.Data.StatMessages)
    args = args + (message,)
    rc = _ProgressMeterUpdate(EasyProgressMeter.Data.MeterID, current_value, EasyProgressMeter.Data.MeterText, *args)
    if current_value >= EasyProgressMeter.Data.MaxValue or not rc:
        EasyProgressMeter.Data.MeterID = None
        del EasyProgressMeter.Data
        EasyProgressMeter.Data = EasyProgressMeterDataClass()
        return False
    return rc

def EasyProgressMeterCancel(title, *args):
    if False:
        print('Hello World!')
    EasyProgressMeter.EasyProgressMeterData = getattr(EasyProgressMeter, 'EasyProgressMeterData', EasyProgressMeterDataClass())
    if EasyProgressMeter.EasyProgressMeterData.MeterID is not None:
        rc = EasyProgressMeter(title, EasyProgressMeter.EasyProgressMeterData.MaxValue, EasyProgressMeter.EasyProgressMeterData.MaxValue, ' *** CANCELLING ***', 'Caller requested a cancel', *args)
        return rc
    return True
_one_line_progress_meters = {}

def OneLineProgressMeter(title, current_value, max_value, key='OK for 1 meter', *args, orientation=None, bar_color=(None, None), button_color=None, size=DEFAULT_PROGRESS_BAR_SIZE, border_width=None, grab_anywhere=False):
    if False:
        print('Hello World!')
    global _one_line_progress_meters
    local_border_width = DEFAULT_PROGRESS_BAR_BORDER_WIDTH if border_width is not None else border_width
    try:
        meter_data = _one_line_progress_meters[key]
    except:
        if int(current_value) >= int(max_value):
            return False
        meter_data = EasyProgressMeterDataClass(title, 1, int(max_value), datetime.datetime.utcnow(), [])
        _one_line_progress_meters[key] = meter_data
        meter_data.ComputeProgressStats()
        message = '\n'.join([line for line in meter_data.StatMessages])
        (meter_data.MeterID, meter_data.MeterText) = _ProgressMeter(title, int(max_value), message, *args, orientation=orientation, bar_color=bar_color, size=size, button_color=button_color, border_width=local_border_width, grab_anywhere=grab_anywhere)
        meter_data.ParentForm = meter_data.MeterID.ParentForm
        return True
    if meter_data.MaxValue == max_value and meter_data.CurrentValue == current_value:
        return True
    meter_data.CurrentValue = int(current_value)
    meter_data.MaxValue = int(max_value)
    meter_data.ComputeProgressStats()
    message = ''
    for line in meter_data.StatMessages:
        message = message + str(line) + '\n'
    message = '\n'.join(meter_data.StatMessages)
    args = args + (message,)
    rc = _ProgressMeterUpdate(meter_data.MeterID, current_value, meter_data.MeterText, *args)
    if current_value >= meter_data.MaxValue or not rc:
        del _one_line_progress_meters[key]
        return False
    return rc

def OneLineProgressMeterCancel(key='OK for 1 meter'):
    if False:
        print('Hello World!')
    global _one_line_progress_meters
    try:
        meter_data = _one_line_progress_meters[key]
    except:
        return
    OneLineProgressMeter('', meter_data.MaxValue, meter_data.MaxValue, key=key)

def GetComplimentaryHex(color):
    if False:
        print('Hello World!')
    color = color[1:]
    color = int(color, 16)
    comp_color = 16777215 ^ color
    comp_color = '#%06X' % comp_color
    return comp_color
_easy_print_data = None

class DebugWin:

    def __init__(self, size=(None, None), location=(None, None), font=None, no_titlebar=False, no_button=False, grab_anywhere=False, keep_on_top=False):
        if False:
            for i in range(10):
                print('nop')
        win_size = size if size != (None, None) else DEFAULT_DEBUG_WINDOW_SIZE
        self.window = Window('Debug Window', no_titlebar=no_titlebar, auto_size_text=True, location=location, font=font or ('Courier New', 10), grab_anywhere=grab_anywhere, keep_on_top=keep_on_top)
        self.output_element = Output(size=win_size)
        if no_button:
            self.layout = [[self.output_element]]
        else:
            self.layout = [[self.output_element], [DummyButton('Quit')]]
        self.window.AddRows(self.layout)
        self.window.Read(timeout=0)
        return

    def Print(self, *args, end=None, sep=None):
        if False:
            while True:
                i = 10
        sepchar = sep if sep is not None else ' '
        endchar = end if end is not None else '\n'
        if self.window is None:
            print(*args, sep=sepchar, end=endchar)
            return
        (event, values) = self.window.Read(timeout=0)
        if event == 'Quit' or event is None:
            self.Close()
        print(*args, sep=sepchar, end=endchar)
        try:
            state = self.window.TKroot.state()
        except:
            self.Close()

    def Close(self):
        if False:
            while True:
                i = 10
        self.window.Close()
        self.window = None

def PrintClose():
    if False:
        return 10
    EasyPrintClose()

def EasyPrint(*args, size=(None, None), end=None, sep=None, location=(None, None), font=None, no_titlebar=False, no_button=False, grab_anywhere=False, keep_on_top=False):
    if False:
        return 10
    global _easy_print_data
    if _easy_print_data is None:
        _easy_print_data = DebugWin(size=size, location=location, font=font, no_titlebar=no_titlebar, no_button=no_button, grab_anywhere=grab_anywhere, keep_on_top=keep_on_top)
    _easy_print_data.Print(*args, end=end, sep=sep)
Print = EasyPrint
eprint = EasyPrint

def EasyPrintClose():
    if False:
        for i in range(10):
            print('nop')
    global _easy_print_data
    if _easy_print_data is not None:
        _easy_print_data.Close()
        _easy_print_data = None
CPRINT_DESTINATION_WINDOW = None
CPRINT_DESTINATION_MULTILINE_ELMENT_KEY = None

def cprint_set_output_destination(window, multiline_key):
    if False:
        print('Hello World!')
    '\n    Sets up the color print (cprint) output destination\n    :param window: The window that the cprint call will route the output to\n    :type window: (Window)\n    :param multiline_key: Key for the Multiline Element where output will be sent\n    :type multiline_key: (Any)\n    :return: None\n    :rtype: None\n    '
    global CPRINT_DESTINATION_WINDOW, CPRINT_DESTINATION_MULTILINE_ELMENT_KEY
    CPRINT_DESTINATION_WINDOW = window
    CPRINT_DESTINATION_MULTILINE_ELMENT_KEY = multiline_key

def cprint(*args, end=None, sep=' ', text_color=None, t=None, background_color=None, b=None, colors=None, c=None, window=None, key=None):
    if False:
        while True:
            i = 10
    '\n    Color print to a multiline element in a window of your choice.\n    Must have EITHER called cprint_set_output_destination prior to making this call so that the\n    window and element key can be saved and used here to route the output, OR used the window\n    and key parameters to the cprint function to specicy these items.\n\n    args is a variable number of things you want to print.\n\n    end - The end char to use just like print uses\n    sep - The separation character like print uses\n    text_color - The color of the text\n            key - overrides the previously defined Multiline key\n    window - overrides the previously defined window to output to\n    background_color - The color of the background\n    colors -(str, str) or str.  A combined text/background color definition in a single parameter\n\n    There are also "aliases" for text_color, background_color and colors (t, b, c)\n    t - An alias for color of the text (makes for shorter calls)\n    b - An alias for the background_color parameter\n    c - Tuple[str, str] - "shorthand" way of specifying color. (foreground, backgrouned)\n    c - str - can also be a string of the format "foreground on background"  ("white on red")\n\n    With the aliases it\'s possible to write the same print but in more compact ways:\n    cprint(\'This will print white text on red background\', c=(\'white\', \'red\'))\n    cprint(\'This will print white text on red background\', c=\'white on red\')\n    cprint(\'This will print white text on red background\', text_color=\'white\', background_color=\'red\')\n    cprint(\'This will print white text on red background\', t=\'white\', b=\'red\')\n\n    :param *args: stuff to output\n    :type *args: (Any)\n    :param text_color: Color of the text\n    :type text_color: (str)\n    :param background_color: The background color of the line\n    :type background_color: (str)\n    :param colors: Either a tuple or a string that has both the text and background colors\n    :type colors: (str) or Tuple[str, str]\n    :param t: Color of the text\n    :type t: (str)\n    :param b: The background color of the line\n    :type b: (str)\n    :param c: Either a tuple or a string that has both the text and background colors\n    :type c: (str) or Tuple[str, str]\n    :param end: end character\n    :type end: (str)\n    :param sep: separator character\n    :type sep: (str)\n    :param key: key of multiline to output to (if you want to override the one previously set)\n    :type key: (Any)\n    :param window: Window containing the multiline to output to (if you want to override the one previously set)\n    :type window: (Window)\n    :return: None\n    :rtype: None\n    '
    destination_key = CPRINT_DESTINATION_MULTILINE_ELMENT_KEY if key is None else key
    destination_window = window or CPRINT_DESTINATION_WINDOW
    if destination_window is None and window is None or (destination_key is None and key is None):
        print('** Warning ** Attempting to perform a cprint without a valid window & key', 'Will instead print on Console', 'You can specify window and key in this cprint call, or set ahead of time using cprint_set_output_destination')
        print(*args)
        return
    kw_text_color = text_color or t
    kw_background_color = background_color or b
    dual_color = colors or c
    try:
        if isinstance(dual_color, tuple):
            kw_text_color = dual_color[0]
            kw_background_color = dual_color[1]
        elif isinstance(dual_color, str):
            kw_text_color = dual_color.split(' on ')[0]
            kw_background_color = dual_color.split(' on ')[1]
    except Exception as e:
        print('* cprint warning * you messed up with color formatting', e)
    mline = destination_window.find_element(destination_key, silent_on_error=True)
    try:
        if end is None:
            mline.print(*args, text_color=kw_text_color, background_color=kw_background_color, end='', sep=sep)
            mline.print('')
        else:
            mline.print(*args, text_color=kw_text_color, background_color=kw_background_color, end=end, sep=sep)
    except Exception as e:
        print('** cprint error trying to print to the multiline. Printing to console instead **', e)
        print(*args, end=end, sep=sep)

def _print_to_element(multiline_element, *args, end=None, sep=None, text_color=None, background_color=None, autoscroll=True):
    if False:
        for i in range(10):
            print('nop')
    '\n    Print like Python normally prints except route the output to a multline element and also add colors if desired\n\n    :param multiline_element:  The multiline element to be output to\n    :type multiline_element: Multiline or MultilineOutput\n    :param args:  The arguments to print\n    :type args: List[Any]\n    :param end:  The end char to use just like print uses\n    :type end: (str)\n    :param sep:  The separation character like print uses\n    :type sep: (str)\n    :param text_color: color of the text\n    :type text_color: (str)\n    :param background_color: The background color of the line\n    :type background_color: (str)\n    :param autoscroll: If True (the default), the element will scroll to bottom after updating\n    :type autoscroll: Bool\n    '
    end_str = str(end) if end is not None else '\n'
    sep_str = str(sep) if sep is not None else ' '
    outstring = ''
    num_args = len(args)
    for (i, arg) in enumerate(args):
        outstring += str(arg)
        if i != num_args - 1:
            outstring += sep_str
    outstring += end_str
    multiline_element.update(outstring, append=True, text_color=text_color, background_color=background_color, autoscroll=autoscroll)

def PopupScrolled(*args, button_color=None, yes_no=False, auto_close=False, auto_close_duration=None, size=(None, None)):
    if False:
        print('Hello World!')
    if not args:
        return
    (width, height) = size
    width = width if width else MESSAGE_BOX_LINE_WIDTH
    form = Window(args[0], auto_size_text=True, button_color=button_color, auto_close=auto_close, auto_close_duration=auto_close_duration)
    (max_line_total, max_line_width, total_lines, height_computed) = (0, 0, 0, 0)
    complete_output = ''
    for message in args:
        message = str(message)
        longest_line_len = max([len(l) for l in message.split('\n')])
        width_used = min(longest_line_len, width)
        max_line_total = max(max_line_total, width_used)
        max_line_width = width
        lines_needed = _GetNumLinesNeeded(message, width_used)
        height_computed += lines_needed
        complete_output += message + '\n'
        total_lines += lines_needed
    height_computed = MAX_SCROLLED_TEXT_BOX_HEIGHT if height_computed > MAX_SCROLLED_TEXT_BOX_HEIGHT else height_computed
    if height:
        height_computed = height
    form.AddRow(Multiline(complete_output, size=(max_line_width, height_computed)))
    pad = max_line_total - 15 if max_line_total > 15 else 1
    if yes_no:
        form.AddRow(Text('', size=(pad, 1), auto_size_text=False), Yes(), No())
        (button, values) = form.Read()
        return button
    else:
        form.AddRow(Text('', size=(pad, 1), auto_size_text=False), Button('OK', size=(5, 1), button_color=button_color))
    (button, values) = form.Read()
    form.Close()
    return button
ScrolledTextBox = PopupScrolled

def SetGlobalIcon(icon):
    if False:
        i = 10
        return i + 15
    global _my_windows
    try:
        with open(icon, 'r') as icon_file:
            pass
    except:
        raise FileNotFoundError
    _my_windows.user_defined_icon = icon
    return True

def SetOptions(icon=None, button_color=None, element_size=(None, None), button_element_size=(None, None), margins=(None, None), element_padding=(None, None), auto_size_text=None, auto_size_buttons=None, font=None, border_width=None, slider_border_width=None, slider_relief=None, slider_orientation=None, autoclose_time=None, message_box_line_width=None, progress_meter_border_depth=None, progress_meter_style=None, progress_meter_relief=None, progress_meter_color=None, progress_meter_size=None, text_justification=None, background_color=None, element_background_color=None, text_element_background_color=None, input_elements_background_color=None, input_text_color=None, scrollbar_color=None, text_color=None, element_text_color=None, debug_win_size=(None, None), window_location=(None, None), tooltip_time=None):
    if False:
        print('Hello World!')
    global DEFAULT_ELEMENT_SIZE
    global DEFAULT_BUTTON_ELEMENT_SIZE
    global DEFAULT_MARGINS
    global DEFAULT_ELEMENT_PADDING
    global DEFAULT_AUTOSIZE_TEXT
    global DEFAULT_AUTOSIZE_BUTTONS
    global DEFAULT_FONT
    global DEFAULT_BORDER_WIDTH
    global DEFAULT_AUTOCLOSE_TIME
    global DEFAULT_BUTTON_COLOR
    global MESSAGE_BOX_LINE_WIDTH
    global DEFAULT_PROGRESS_BAR_BORDER_WIDTH
    global DEFAULT_PROGRESS_BAR_STYLE
    global DEFAULT_PROGRESS_BAR_RELIEF
    global DEFAULT_PROGRESS_BAR_COLOR
    global DEFAULT_PROGRESS_BAR_SIZE
    global DEFAULT_TEXT_JUSTIFICATION
    global DEFAULT_DEBUG_WINDOW_SIZE
    global DEFAULT_SLIDER_BORDER_WIDTH
    global DEFAULT_SLIDER_RELIEF
    global DEFAULT_SLIDER_ORIENTATION
    global DEFAULT_BACKGROUND_COLOR
    global DEFAULT_INPUT_ELEMENTS_COLOR
    global DEFAULT_ELEMENT_BACKGROUND_COLOR
    global DEFAULT_TEXT_ELEMENT_BACKGROUND_COLOR
    global DEFAULT_SCROLLBAR_COLOR
    global DEFAULT_TEXT_COLOR
    global DEFAULT_WINDOW_LOCATION
    global DEFAULT_ELEMENT_TEXT_COLOR
    global DEFAULT_INPUT_TEXT_COLOR
    global DEFAULT_TOOLTIP_TIME
    global _my_windows
    if icon:
        try:
            with open(icon, 'r') as icon_file:
                pass
        except:
            raise FileNotFoundError
        _my_windows.user_defined_icon = icon
    if button_color != None:
        DEFAULT_BUTTON_COLOR = button_color
    if element_size != (None, None):
        DEFAULT_ELEMENT_SIZE = element_size
    if button_element_size != (None, None):
        DEFAULT_BUTTON_ELEMENT_SIZE = button_element_size
    if margins != (None, None):
        DEFAULT_MARGINS = margins
    if element_padding != (None, None):
        DEFAULT_ELEMENT_PADDING = element_padding
    if auto_size_text != None:
        DEFAULT_AUTOSIZE_TEXT = auto_size_text
    if auto_size_buttons != None:
        DEFAULT_AUTOSIZE_BUTTONS = auto_size_buttons
    if font != None:
        DEFAULT_FONT = font
    if border_width != None:
        DEFAULT_BORDER_WIDTH = border_width
    if autoclose_time != None:
        DEFAULT_AUTOCLOSE_TIME = autoclose_time
    if message_box_line_width != None:
        MESSAGE_BOX_LINE_WIDTH = message_box_line_width
    if progress_meter_border_depth != None:
        DEFAULT_PROGRESS_BAR_BORDER_WIDTH = progress_meter_border_depth
    if progress_meter_style != None:
        DEFAULT_PROGRESS_BAR_STYLE = progress_meter_style
    if progress_meter_relief != None:
        DEFAULT_PROGRESS_BAR_RELIEF = progress_meter_relief
    if progress_meter_color != None:
        DEFAULT_PROGRESS_BAR_COLOR = progress_meter_color
    if progress_meter_size != None:
        DEFAULT_PROGRESS_BAR_SIZE = progress_meter_size
    if slider_border_width != None:
        DEFAULT_SLIDER_BORDER_WIDTH = slider_border_width
    if slider_orientation != None:
        DEFAULT_SLIDER_ORIENTATION = slider_orientation
    if slider_relief != None:
        DEFAULT_SLIDER_RELIEF = slider_relief
    if text_justification != None:
        DEFAULT_TEXT_JUSTIFICATION = text_justification
    if background_color != None:
        DEFAULT_BACKGROUND_COLOR = background_color
    if text_element_background_color != None:
        DEFAULT_TEXT_ELEMENT_BACKGROUND_COLOR = text_element_background_color
    if input_elements_background_color != None:
        DEFAULT_INPUT_ELEMENTS_COLOR = input_elements_background_color
    if element_background_color != None:
        DEFAULT_ELEMENT_BACKGROUND_COLOR = element_background_color
    if window_location != (None, None):
        DEFAULT_WINDOW_LOCATION = window_location
    if debug_win_size != (None, None):
        DEFAULT_DEBUG_WINDOW_SIZE = debug_win_size
    if text_color != None:
        DEFAULT_TEXT_COLOR = text_color
    if scrollbar_color != None:
        DEFAULT_SCROLLBAR_COLOR = scrollbar_color
    if element_text_color != None:
        DEFAULT_ELEMENT_TEXT_COLOR = element_text_color
    if input_text_color is not None:
        DEFAULT_INPUT_TEXT_COLOR = input_text_color
    if tooltip_time is not None:
        DEFAULT_TOOLTIP_TIME = tooltip_time
    return True
LOOK_AND_FEEL_TABLE = {'SystemDefault': {'BACKGROUND': COLOR_SYSTEM_DEFAULT, 'TEXT': COLOR_SYSTEM_DEFAULT, 'INPUT': COLOR_SYSTEM_DEFAULT, 'TEXT_INPUT': COLOR_SYSTEM_DEFAULT, 'SCROLL': COLOR_SYSTEM_DEFAULT, 'BUTTON': OFFICIAL_PYSIMPLEGUI_BUTTON_COLOR, 'PROGRESS': COLOR_SYSTEM_DEFAULT, 'BORDER': 1, 'SLIDER_DEPTH': 1, 'PROGRESS_DEPTH': 0}, 'SystemDefaultForReal': {'BACKGROUND': COLOR_SYSTEM_DEFAULT, 'TEXT': COLOR_SYSTEM_DEFAULT, 'INPUT': COLOR_SYSTEM_DEFAULT, 'TEXT_INPUT': COLOR_SYSTEM_DEFAULT, 'SCROLL': COLOR_SYSTEM_DEFAULT, 'BUTTON': COLOR_SYSTEM_DEFAULT, 'PROGRESS': COLOR_SYSTEM_DEFAULT, 'BORDER': 1, 'SLIDER_DEPTH': 1, 'PROGRESS_DEPTH': 0}, 'SystemDefault1': {'BACKGROUND': COLOR_SYSTEM_DEFAULT, 'TEXT': COLOR_SYSTEM_DEFAULT, 'INPUT': COLOR_SYSTEM_DEFAULT, 'TEXT_INPUT': COLOR_SYSTEM_DEFAULT, 'SCROLL': COLOR_SYSTEM_DEFAULT, 'BUTTON': COLOR_SYSTEM_DEFAULT, 'PROGRESS': COLOR_SYSTEM_DEFAULT, 'BORDER': 1, 'SLIDER_DEPTH': 1, 'PROGRESS_DEPTH': 0}, 'Material1': {'BACKGROUND': '#E3F2FD', 'TEXT': '#000000', 'INPUT': '#86A8FF', 'TEXT_INPUT': '#000000', 'SCROLL': '#86A8FF', 'BUTTON': ('#FFFFFF', '#5079D3'), 'PROGRESS': DEFAULT_PROGRESS_BAR_COLOR, 'BORDER': 0, 'SLIDER_DEPTH': 0, 'PROGRESS_DEPTH': 0, 'ACCENT1': '#FF0266', 'ACCENT2': '#FF5C93', 'ACCENT3': '#C5003C'}, 'Material2': {'BACKGROUND': '#FAFAFA', 'TEXT': '#000000', 'INPUT': '#004EA1', 'TEXT_INPUT': '#FFFFFF', 'SCROLL': '#5EA7FF', 'BUTTON': ('#FFFFFF', '#0079D3'), 'PROGRESS': DEFAULT_PROGRESS_BAR_COLOR, 'BORDER': 0, 'SLIDER_DEPTH': 0, 'PROGRESS_DEPTH': 0, 'ACCENT1': '#FF0266', 'ACCENT2': '#FF5C93', 'ACCENT3': '#C5003C'}, 'Reddit': {'BACKGROUND': '#ffffff', 'TEXT': '#1a1a1b', 'INPUT': '#dae0e6', 'TEXT_INPUT': '#222222', 'SCROLL': '#a5a4a4', 'BUTTON': ('#FFFFFF', '#0079d3'), 'PROGRESS': DEFAULT_PROGRESS_BAR_COLOR, 'BORDER': 1, 'SLIDER_DEPTH': 0, 'PROGRESS_DEPTH': 0, 'ACCENT1': '#ff5414', 'ACCENT2': '#33a8ff', 'ACCENT3': '#dbf0ff'}, 'Topanga': {'BACKGROUND': '#282923', 'TEXT': '#E7DB74', 'INPUT': '#393a32', 'TEXT_INPUT': '#E7C855', 'SCROLL': '#E7C855', 'BUTTON': ('#E7C855', '#284B5A'), 'PROGRESS': DEFAULT_PROGRESS_BAR_COLOR, 'BORDER': 1, 'SLIDER_DEPTH': 0, 'PROGRESS_DEPTH': 0, 'ACCENT1': '#c15226', 'ACCENT2': '#7a4d5f', 'ACCENT3': '#889743'}, 'GreenTan': {'BACKGROUND': '#9FB8AD', 'TEXT': COLOR_SYSTEM_DEFAULT, 'INPUT': '#F7F3EC', 'TEXT_INPUT': '#000000', 'SCROLL': '#F7F3EC', 'BUTTON': ('#FFFFFF', '#475841'), 'PROGRESS': DEFAULT_PROGRESS_BAR_COLOR, 'BORDER': 1, 'SLIDER_DEPTH': 0, 'PROGRESS_DEPTH': 0}, 'Dark': {'BACKGROUND': '#404040', 'TEXT': '#FFFFFF', 'INPUT': '#4D4D4D', 'TEXT_INPUT': '#FFFFFF', 'SCROLL': '#707070', 'BUTTON': ('#FFFFFF', '#004F00'), 'PROGRESS': DEFAULT_PROGRESS_BAR_COLOR, 'BORDER': 1, 'SLIDER_DEPTH': 0, 'PROGRESS_DEPTH': 0}, 'LightGreen': {'BACKGROUND': '#B7CECE', 'TEXT': '#000000', 'INPUT': '#FDFFF7', 'TEXT_INPUT': '#000000', 'SCROLL': '#FDFFF7', 'BUTTON': ('#FFFFFF', '#658268'), 'PROGRESS': DEFAULT_PROGRESS_BAR_COLOR, 'BORDER': 1, 'SLIDER_DEPTH': 0, 'ACCENT1': '#76506d', 'ACCENT2': '#5148f1', 'ACCENT3': '#0a1c84', 'PROGRESS_DEPTH': 0}, 'Dark2': {'BACKGROUND': '#404040', 'TEXT': '#FFFFFF', 'INPUT': '#FFFFFF', 'TEXT_INPUT': '#000000', 'SCROLL': '#707070', 'BUTTON': ('#FFFFFF', '#004F00'), 'PROGRESS': DEFAULT_PROGRESS_BAR_COLOR, 'BORDER': 1, 'SLIDER_DEPTH': 0, 'PROGRESS_DEPTH': 0}, 'Black': {'BACKGROUND': '#000000', 'TEXT': '#FFFFFF', 'INPUT': '#4D4D4D', 'TEXT_INPUT': '#FFFFFF', 'SCROLL': '#707070', 'BUTTON': ('#000000', '#FFFFFF'), 'PROGRESS': DEFAULT_PROGRESS_BAR_COLOR, 'BORDER': 1, 'SLIDER_DEPTH': 0, 'PROGRESS_DEPTH': 0}, 'Tan': {'BACKGROUND': '#fdf6e3', 'TEXT': '#268bd1', 'INPUT': '#eee8d5', 'TEXT_INPUT': '#6c71c3', 'SCROLL': '#eee8d5', 'BUTTON': ('#FFFFFF', '#063542'), 'PROGRESS': DEFAULT_PROGRESS_BAR_COLOR, 'BORDER': 1, 'SLIDER_DEPTH': 0, 'PROGRESS_DEPTH': 0}, 'TanBlue': {'BACKGROUND': '#e5dece', 'TEXT': '#063289', 'INPUT': '#f9f8f4', 'TEXT_INPUT': '#242834', 'SCROLL': '#eee8d5', 'BUTTON': ('#FFFFFF', '#063289'), 'PROGRESS': DEFAULT_PROGRESS_BAR_COLOR, 'BORDER': 1, 'SLIDER_DEPTH': 0, 'PROGRESS_DEPTH': 0}, 'DarkTanBlue': {'BACKGROUND': '#242834', 'TEXT': '#dfe6f8', 'INPUT': '#97755c', 'TEXT_INPUT': '#FFFFFF', 'SCROLL': '#a9afbb', 'BUTTON': ('#FFFFFF', '#063289'), 'PROGRESS': DEFAULT_PROGRESS_BAR_COLOR, 'BORDER': 1, 'SLIDER_DEPTH': 0, 'PROGRESS_DEPTH': 0}, 'DarkAmber': {'BACKGROUND': '#2c2825', 'TEXT': '#fdcb52', 'INPUT': '#705e52', 'TEXT_INPUT': '#fdcb52', 'SCROLL': '#705e52', 'BUTTON': ('#000000', '#fdcb52'), 'PROGRESS': DEFAULT_PROGRESS_BAR_COLOR, 'BORDER': 1, 'SLIDER_DEPTH': 0, 'PROGRESS_DEPTH': 0}, 'DarkBlue': {'BACKGROUND': '#1a2835', 'TEXT': '#d1ecff', 'INPUT': '#335267', 'TEXT_INPUT': '#acc2d0', 'SCROLL': '#1b6497', 'BUTTON': ('#000000', '#fafaf8'), 'PROGRESS': DEFAULT_PROGRESS_BAR_COLOR, 'BORDER': 1, 'SLIDER_DEPTH': 0, 'PROGRESS_DEPTH': 0}, 'Reds': {'BACKGROUND': '#280001', 'TEXT': '#FFFFFF', 'INPUT': '#d8d584', 'TEXT_INPUT': '#000000', 'SCROLL': '#763e00', 'BUTTON': ('#000000', '#daad28'), 'PROGRESS': DEFAULT_PROGRESS_BAR_COLOR, 'BORDER': 1, 'SLIDER_DEPTH': 0, 'PROGRESS_DEPTH': 0}, 'Green': {'BACKGROUND': '#82a459', 'TEXT': '#000000', 'INPUT': '#d8d584', 'TEXT_INPUT': '#000000', 'SCROLL': '#e3ecf3', 'BUTTON': ('#FFFFFF', '#517239'), 'PROGRESS': DEFAULT_PROGRESS_BAR_COLOR, 'BORDER': 1, 'SLIDER_DEPTH': 0, 'PROGRESS_DEPTH': 0}, 'BluePurple': {'BACKGROUND': '#A5CADD', 'TEXT': '#6E266E', 'INPUT': '#E0F5FF', 'TEXT_INPUT': '#000000', 'SCROLL': '#E0F5FF', 'BUTTON': ('#FFFFFF', '#303952'), 'PROGRESS': DEFAULT_PROGRESS_BAR_COLOR, 'BORDER': 1, 'SLIDER_DEPTH': 0, 'PROGRESS_DEPTH': 0}, 'Purple': {'BACKGROUND': '#B0AAC2', 'TEXT': '#000000', 'INPUT': '#F2EFE8', 'SCROLL': '#F2EFE8', 'TEXT_INPUT': '#000000', 'BUTTON': ('#000000', '#C2D4D8'), 'PROGRESS': DEFAULT_PROGRESS_BAR_COLOR, 'BORDER': 1, 'SLIDER_DEPTH': 0, 'PROGRESS_DEPTH': 0}, 'BlueMono': {'BACKGROUND': '#AAB6D3', 'TEXT': '#000000', 'INPUT': '#F1F4FC', 'SCROLL': '#F1F4FC', 'TEXT_INPUT': '#000000', 'BUTTON': ('#FFFFFF', '#7186C7'), 'PROGRESS': DEFAULT_PROGRESS_BAR_COLOR, 'BORDER': 1, 'SLIDER_DEPTH': 0, 'PROGRESS_DEPTH': 0}, 'GreenMono': {'BACKGROUND': '#A8C1B4', 'TEXT': '#000000', 'INPUT': '#DDE0DE', 'SCROLL': '#E3E3E3', 'TEXT_INPUT': '#000000', 'BUTTON': ('#FFFFFF', '#6D9F85'), 'PROGRESS': DEFAULT_PROGRESS_BAR_COLOR, 'BORDER': 1, 'SLIDER_DEPTH': 0, 'PROGRESS_DEPTH': 0}, 'BrownBlue': {'BACKGROUND': '#64778d', 'TEXT': '#FFFFFF', 'INPUT': '#f0f3f7', 'SCROLL': '#A6B2BE', 'TEXT_INPUT': '#000000', 'BUTTON': ('#FFFFFF', '#283b5b'), 'PROGRESS': DEFAULT_PROGRESS_BAR_COLOR, 'BORDER': 1, 'SLIDER_DEPTH': 0, 'PROGRESS_DEPTH': 0}, 'BrightColors': {'BACKGROUND': '#b4ffb4', 'TEXT': '#000000', 'INPUT': '#ffff64', 'SCROLL': '#ffb482', 'TEXT_INPUT': '#000000', 'BUTTON': ('#000000', '#ffa0dc'), 'PROGRESS': DEFAULT_PROGRESS_BAR_COLOR, 'BORDER': 1, 'SLIDER_DEPTH': 0, 'PROGRESS_DEPTH': 0}, 'NeutralBlue': {'BACKGROUND': '#92aa9d', 'TEXT': '#000000', 'INPUT': '#fcfff6', 'SCROLL': '#fcfff6', 'TEXT_INPUT': '#000000', 'BUTTON': ('#000000', '#d0dbbd'), 'PROGRESS': DEFAULT_PROGRESS_BAR_COLOR, 'BORDER': 1, 'SLIDER_DEPTH': 0, 'PROGRESS_DEPTH': 0}, 'Kayak': {'BACKGROUND': '#a7ad7f', 'TEXT': '#000000', 'INPUT': '#e6d3a8', 'SCROLL': '#e6d3a8', 'TEXT_INPUT': '#000000', 'BUTTON': ('#FFFFFF', '#5d907d'), 'PROGRESS': DEFAULT_PROGRESS_BAR_COLOR, 'BORDER': 1, 'SLIDER_DEPTH': 0, 'PROGRESS_DEPTH': 0}, 'SandyBeach': {'BACKGROUND': '#efeccb', 'TEXT': '#012f2f', 'INPUT': '#e6d3a8', 'SCROLL': '#e6d3a8', 'TEXT_INPUT': '#012f2f', 'BUTTON': ('#FFFFFF', '#046380'), 'PROGRESS': DEFAULT_PROGRESS_BAR_COLOR, 'BORDER': 1, 'SLIDER_DEPTH': 0, 'PROGRESS_DEPTH': 0}, 'TealMono': {'BACKGROUND': '#a8cfdd', 'TEXT': '#000000', 'INPUT': '#dfedf2', 'SCROLL': '#dfedf2', 'TEXT_INPUT': '#000000', 'BUTTON': ('#FFFFFF', '#183440'), 'PROGRESS': DEFAULT_PROGRESS_BAR_COLOR, 'BORDER': 1, 'SLIDER_DEPTH': 0, 'PROGRESS_DEPTH': 0}, 'Default': {'BACKGROUND': COLOR_SYSTEM_DEFAULT, 'TEXT': COLOR_SYSTEM_DEFAULT, 'INPUT': COLOR_SYSTEM_DEFAULT, 'TEXT_INPUT': COLOR_SYSTEM_DEFAULT, 'SCROLL': COLOR_SYSTEM_DEFAULT, 'BUTTON': OFFICIAL_PYSIMPLEGUI_BUTTON_COLOR, 'PROGRESS': COLOR_SYSTEM_DEFAULT, 'BORDER': 1, 'SLIDER_DEPTH': 1, 'PROGRESS_DEPTH': 0}, 'Default1': {'BACKGROUND': COLOR_SYSTEM_DEFAULT, 'TEXT': COLOR_SYSTEM_DEFAULT, 'INPUT': COLOR_SYSTEM_DEFAULT, 'TEXT_INPUT': COLOR_SYSTEM_DEFAULT, 'SCROLL': COLOR_SYSTEM_DEFAULT, 'BUTTON': COLOR_SYSTEM_DEFAULT, 'PROGRESS': COLOR_SYSTEM_DEFAULT, 'BORDER': 1, 'SLIDER_DEPTH': 1, 'PROGRESS_DEPTH': 0}, 'DefaultNoMoreNagging': {'BACKGROUND': COLOR_SYSTEM_DEFAULT, 'TEXT': COLOR_SYSTEM_DEFAULT, 'INPUT': COLOR_SYSTEM_DEFAULT, 'TEXT_INPUT': COLOR_SYSTEM_DEFAULT, 'SCROLL': COLOR_SYSTEM_DEFAULT, 'BUTTON': OFFICIAL_PYSIMPLEGUI_BUTTON_COLOR, 'PROGRESS': COLOR_SYSTEM_DEFAULT, 'BORDER': 1, 'SLIDER_DEPTH': 1, 'PROGRESS_DEPTH': 0}, 'LightBlue': {'BACKGROUND': '#E3F2FD', 'TEXT': '#000000', 'INPUT': '#86A8FF', 'TEXT_INPUT': '#000000', 'SCROLL': '#86A8FF', 'BUTTON': ('#FFFFFF', '#5079D3'), 'PROGRESS': DEFAULT_PROGRESS_BAR_COLOR, 'BORDER': 0, 'SLIDER_DEPTH': 0, 'PROGRESS_DEPTH': 0, 'ACCENT1': '#FF0266', 'ACCENT2': '#FF5C93', 'ACCENT3': '#C5003C'}, 'LightGrey': {'BACKGROUND': '#FAFAFA', 'TEXT': '#000000', 'INPUT': '#004EA1', 'TEXT_INPUT': '#FFFFFF', 'SCROLL': '#5EA7FF', 'BUTTON': ('#FFFFFF', '#0079D3'), 'PROGRESS': DEFAULT_PROGRESS_BAR_COLOR, 'BORDER': 0, 'SLIDER_DEPTH': 0, 'PROGRESS_DEPTH': 0, 'ACCENT1': '#FF0266', 'ACCENT2': '#FF5C93', 'ACCENT3': '#C5003C'}, 'LightGrey1': {'BACKGROUND': '#ffffff', 'TEXT': '#1a1a1b', 'INPUT': '#dae0e6', 'TEXT_INPUT': '#222222', 'SCROLL': '#a5a4a4', 'BUTTON': ('#FFFFFF', '#0079d3'), 'PROGRESS': DEFAULT_PROGRESS_BAR_COLOR, 'BORDER': 1, 'SLIDER_DEPTH': 0, 'PROGRESS_DEPTH': 0, 'ACCENT1': '#ff5414', 'ACCENT2': '#33a8ff', 'ACCENT3': '#dbf0ff'}, 'DarkBrown': {'BACKGROUND': '#282923', 'TEXT': '#E7DB74', 'INPUT': '#393a32', 'TEXT_INPUT': '#E7C855', 'SCROLL': '#E7C855', 'BUTTON': ('#E7C855', '#284B5A'), 'PROGRESS': DEFAULT_PROGRESS_BAR_COLOR, 'BORDER': 1, 'SLIDER_DEPTH': 0, 'PROGRESS_DEPTH': 0, 'ACCENT1': '#c15226', 'ACCENT2': '#7a4d5f', 'ACCENT3': '#889743'}, 'LightGreen1': {'BACKGROUND': '#9FB8AD', 'TEXT': COLOR_SYSTEM_DEFAULT, 'INPUT': '#F7F3EC', 'TEXT_INPUT': '#000000', 'SCROLL': '#F7F3EC', 'BUTTON': ('#FFFFFF', '#475841'), 'PROGRESS': DEFAULT_PROGRESS_BAR_COLOR, 'BORDER': 1, 'SLIDER_DEPTH': 0, 'PROGRESS_DEPTH': 0}, 'DarkGrey': {'BACKGROUND': '#404040', 'TEXT': '#FFFFFF', 'INPUT': '#4D4D4D', 'TEXT_INPUT': '#FFFFFF', 'SCROLL': '#707070', 'BUTTON': ('#FFFFFF', '#004F00'), 'PROGRESS': DEFAULT_PROGRESS_BAR_COLOR, 'BORDER': 1, 'SLIDER_DEPTH': 0, 'PROGRESS_DEPTH': 0}, 'LightGreen2': {'BACKGROUND': '#B7CECE', 'TEXT': '#000000', 'INPUT': '#FDFFF7', 'TEXT_INPUT': '#000000', 'SCROLL': '#FDFFF7', 'BUTTON': ('#FFFFFF', '#658268'), 'PROGRESS': DEFAULT_PROGRESS_BAR_COLOR, 'BORDER': 1, 'SLIDER_DEPTH': 0, 'ACCENT1': '#76506d', 'ACCENT2': '#5148f1', 'ACCENT3': '#0a1c84', 'PROGRESS_DEPTH': 0}, 'DarkGrey1': {'BACKGROUND': '#404040', 'TEXT': '#FFFFFF', 'INPUT': '#FFFFFF', 'TEXT_INPUT': '#000000', 'SCROLL': '#707070', 'BUTTON': ('#FFFFFF', '#004F00'), 'PROGRESS': DEFAULT_PROGRESS_BAR_COLOR, 'BORDER': 1, 'SLIDER_DEPTH': 0, 'PROGRESS_DEPTH': 0}, 'DarkBlack': {'BACKGROUND': '#000000', 'TEXT': '#FFFFFF', 'INPUT': '#4D4D4D', 'TEXT_INPUT': '#FFFFFF', 'SCROLL': '#707070', 'BUTTON': ('#000000', '#FFFFFF'), 'PROGRESS': DEFAULT_PROGRESS_BAR_COLOR, 'BORDER': 1, 'SLIDER_DEPTH': 0, 'PROGRESS_DEPTH': 0}, 'LightBrown': {'BACKGROUND': '#fdf6e3', 'TEXT': '#268bd1', 'INPUT': '#eee8d5', 'TEXT_INPUT': '#6c71c3', 'SCROLL': '#eee8d5', 'BUTTON': ('#FFFFFF', '#063542'), 'PROGRESS': DEFAULT_PROGRESS_BAR_COLOR, 'BORDER': 1, 'SLIDER_DEPTH': 0, 'PROGRESS_DEPTH': 0}, 'LightBrown1': {'BACKGROUND': '#e5dece', 'TEXT': '#063289', 'INPUT': '#f9f8f4', 'TEXT_INPUT': '#242834', 'SCROLL': '#eee8d5', 'BUTTON': ('#FFFFFF', '#063289'), 'PROGRESS': DEFAULT_PROGRESS_BAR_COLOR, 'BORDER': 1, 'SLIDER_DEPTH': 0, 'PROGRESS_DEPTH': 0}, 'DarkBlue1': {'BACKGROUND': '#242834', 'TEXT': '#dfe6f8', 'INPUT': '#97755c', 'TEXT_INPUT': '#FFFFFF', 'SCROLL': '#a9afbb', 'BUTTON': ('#FFFFFF', '#063289'), 'PROGRESS': DEFAULT_PROGRESS_BAR_COLOR, 'BORDER': 1, 'SLIDER_DEPTH': 0, 'PROGRESS_DEPTH': 0}, 'DarkBrown1': {'BACKGROUND': '#2c2825', 'TEXT': '#fdcb52', 'INPUT': '#705e52', 'TEXT_INPUT': '#fdcb52', 'SCROLL': '#705e52', 'BUTTON': ('#000000', '#fdcb52'), 'PROGRESS': DEFAULT_PROGRESS_BAR_COLOR, 'BORDER': 1, 'SLIDER_DEPTH': 0, 'PROGRESS_DEPTH': 0}, 'DarkBlue2': {'BACKGROUND': '#1a2835', 'TEXT': '#d1ecff', 'INPUT': '#335267', 'TEXT_INPUT': '#acc2d0', 'SCROLL': '#1b6497', 'BUTTON': ('#000000', '#fafaf8'), 'PROGRESS': DEFAULT_PROGRESS_BAR_COLOR, 'BORDER': 1, 'SLIDER_DEPTH': 0, 'PROGRESS_DEPTH': 0}, 'DarkBrown2': {'BACKGROUND': '#280001', 'TEXT': '#FFFFFF', 'INPUT': '#d8d584', 'TEXT_INPUT': '#000000', 'SCROLL': '#763e00', 'BUTTON': ('#000000', '#daad28'), 'PROGRESS': DEFAULT_PROGRESS_BAR_COLOR, 'BORDER': 1, 'SLIDER_DEPTH': 0, 'PROGRESS_DEPTH': 0}, 'DarkGreen': {'BACKGROUND': '#82a459', 'TEXT': '#000000', 'INPUT': '#d8d584', 'TEXT_INPUT': '#000000', 'SCROLL': '#e3ecf3', 'BUTTON': ('#FFFFFF', '#517239'), 'PROGRESS': DEFAULT_PROGRESS_BAR_COLOR, 'BORDER': 1, 'SLIDER_DEPTH': 0, 'PROGRESS_DEPTH': 0}, 'LightBlue1': {'BACKGROUND': '#A5CADD', 'TEXT': '#6E266E', 'INPUT': '#E0F5FF', 'TEXT_INPUT': '#000000', 'SCROLL': '#E0F5FF', 'BUTTON': ('#FFFFFF', '#303952'), 'PROGRESS': DEFAULT_PROGRESS_BAR_COLOR, 'BORDER': 1, 'SLIDER_DEPTH': 0, 'PROGRESS_DEPTH': 0}, 'LightPurple': {'BACKGROUND': '#B0AAC2', 'TEXT': '#000000', 'INPUT': '#F2EFE8', 'SCROLL': '#F2EFE8', 'TEXT_INPUT': '#000000', 'BUTTON': ('#000000', '#C2D4D8'), 'PROGRESS': DEFAULT_PROGRESS_BAR_COLOR, 'BORDER': 1, 'SLIDER_DEPTH': 0, 'PROGRESS_DEPTH': 0}, 'LightBlue2': {'BACKGROUND': '#AAB6D3', 'TEXT': '#000000', 'INPUT': '#F1F4FC', 'SCROLL': '#F1F4FC', 'TEXT_INPUT': '#000000', 'BUTTON': ('#FFFFFF', '#7186C7'), 'PROGRESS': DEFAULT_PROGRESS_BAR_COLOR, 'BORDER': 1, 'SLIDER_DEPTH': 0, 'PROGRESS_DEPTH': 0}, 'LightGreen3': {'BACKGROUND': '#A8C1B4', 'TEXT': '#000000', 'INPUT': '#DDE0DE', 'SCROLL': '#E3E3E3', 'TEXT_INPUT': '#000000', 'BUTTON': ('#FFFFFF', '#6D9F85'), 'PROGRESS': DEFAULT_PROGRESS_BAR_COLOR, 'BORDER': 1, 'SLIDER_DEPTH': 0, 'PROGRESS_DEPTH': 0}, 'DarkBlue3': {'BACKGROUND': '#64778d', 'TEXT': '#FFFFFF', 'INPUT': '#f0f3f7', 'SCROLL': '#A6B2BE', 'TEXT_INPUT': '#000000', 'BUTTON': ('#FFFFFF', '#283b5b'), 'PROGRESS': DEFAULT_PROGRESS_BAR_COLOR, 'BORDER': 1, 'SLIDER_DEPTH': 0, 'PROGRESS_DEPTH': 0}, 'LightGreen4': {'BACKGROUND': '#b4ffb4', 'TEXT': '#000000', 'INPUT': '#ffff64', 'SCROLL': '#ffb482', 'TEXT_INPUT': '#000000', 'BUTTON': ('#000000', '#ffa0dc'), 'PROGRESS': DEFAULT_PROGRESS_BAR_COLOR, 'BORDER': 1, 'SLIDER_DEPTH': 0, 'PROGRESS_DEPTH': 0}, 'LightGreen5': {'BACKGROUND': '#92aa9d', 'TEXT': '#000000', 'INPUT': '#fcfff6', 'SCROLL': '#fcfff6', 'TEXT_INPUT': '#000000', 'BUTTON': ('#000000', '#d0dbbd'), 'PROGRESS': DEFAULT_PROGRESS_BAR_COLOR, 'BORDER': 1, 'SLIDER_DEPTH': 0, 'PROGRESS_DEPTH': 0}, 'LightBrown2': {'BACKGROUND': '#a7ad7f', 'TEXT': '#000000', 'INPUT': '#e6d3a8', 'SCROLL': '#e6d3a8', 'TEXT_INPUT': '#000000', 'BUTTON': ('#FFFFFF', '#5d907d'), 'PROGRESS': DEFAULT_PROGRESS_BAR_COLOR, 'BORDER': 1, 'SLIDER_DEPTH': 0, 'PROGRESS_DEPTH': 0}, 'LightBrown3': {'BACKGROUND': '#efeccb', 'TEXT': '#012f2f', 'INPUT': '#e6d3a8', 'SCROLL': '#e6d3a8', 'TEXT_INPUT': '#012f2f', 'BUTTON': ('#FFFFFF', '#046380'), 'PROGRESS': DEFAULT_PROGRESS_BAR_COLOR, 'BORDER': 1, 'SLIDER_DEPTH': 0, 'PROGRESS_DEPTH': 0}, 'LightBlue3': {'BACKGROUND': '#a8cfdd', 'TEXT': '#000000', 'INPUT': '#dfedf2', 'SCROLL': '#dfedf2', 'TEXT_INPUT': '#000000', 'BUTTON': ('#FFFFFF', '#183440'), 'PROGRESS': DEFAULT_PROGRESS_BAR_COLOR, 'BORDER': 1, 'SLIDER_DEPTH': 0, 'PROGRESS_DEPTH': 0}, 'LightBrown4': {'BACKGROUND': '#d7c79e', 'TEXT': '#a35638', 'INPUT': '#9dab86', 'TEXT_INPUT': '#000000', 'SCROLL': '#a35638', 'BUTTON': ('#FFFFFF', '#a35638'), 'PROGRESS': DEFAULT_PROGRESS_BAR_COLOR, 'BORDER': 1, 'SLIDER_DEPTH': 0, 'PROGRESS_DEPTH': 0, 'COLOR_LIST': ['#a35638', '#9dab86', '#e08f62', '#d7c79e']}, 'DarkTeal': {'BACKGROUND': '#003f5c', 'TEXT': '#fb5b5a', 'INPUT': '#bc4873', 'TEXT_INPUT': '#FFFFFF', 'SCROLL': '#bc4873', 'BUTTON': ('#FFFFFF', '#fb5b5a'), 'PROGRESS': DEFAULT_PROGRESS_BAR_COLOR, 'BORDER': 1, 'SLIDER_DEPTH': 0, 'PROGRESS_DEPTH': 0, 'COLOR_LIST': ['#003f5c', '#472b62', '#bc4873', '#fb5b5a']}, 'DarkPurple': {'BACKGROUND': '#472b62', 'TEXT': '#fb5b5a', 'INPUT': '#bc4873', 'TEXT_INPUT': '#FFFFFF', 'SCROLL': '#bc4873', 'BUTTON': ('#FFFFFF', '#472b62'), 'PROGRESS': DEFAULT_PROGRESS_BAR_COLOR, 'BORDER': 1, 'SLIDER_DEPTH': 0, 'PROGRESS_DEPTH': 0, 'COLOR_LIST': ['#003f5c', '#472b62', '#bc4873', '#fb5b5a']}, 'LightGreen6': {'BACKGROUND': '#eafbea', 'TEXT': '#1f6650', 'INPUT': '#6f9a8d', 'TEXT_INPUT': '#FFFFFF', 'SCROLL': '#1f6650', 'BUTTON': ('#FFFFFF', '#1f6650'), 'PROGRESS': DEFAULT_PROGRESS_BAR_COLOR, 'BORDER': 1, 'SLIDER_DEPTH': 0, 'PROGRESS_DEPTH': 0, 'COLOR_LIST': ['#1f6650', '#6f9a8d', '#ea5e5e', '#eafbea']}, 'DarkGrey2': {'BACKGROUND': '#2b2b28', 'TEXT': '#f8f8f8', 'INPUT': '#f1d6ab', 'TEXT_INPUT': '#000000', 'SCROLL': '#f1d6ab', 'BUTTON': ('#2b2b28', '#e3b04b'), 'PROGRESS': DEFAULT_PROGRESS_BAR_COLOR, 'BORDER': 1, 'SLIDER_DEPTH': 0, 'PROGRESS_DEPTH': 0, 'COLOR_LIST': ['#2b2b28', '#e3b04b', '#f1d6ab', '#f8f8f8']}, 'LightBrown6': {'BACKGROUND': '#f9b282', 'TEXT': '#8f4426', 'INPUT': '#de6b35', 'TEXT_INPUT': '#FFFFFF', 'SCROLL': '#8f4426', 'BUTTON': ('#FFFFFF', '#8f4426'), 'PROGRESS': DEFAULT_PROGRESS_BAR_COLOR, 'BORDER': 1, 'SLIDER_DEPTH': 0, 'PROGRESS_DEPTH': 0, 'COLOR_LIST': ['#8f4426', '#de6b35', '#64ccda', '#f9b282']}, 'DarkTeal1': {'BACKGROUND': '#396362', 'TEXT': '#ffe7d1', 'INPUT': '#f6c89f', 'TEXT_INPUT': '#000000', 'SCROLL': '#f6c89f', 'BUTTON': ('#ffe7d1', '#4b8e8d'), 'PROGRESS': DEFAULT_PROGRESS_BAR_COLOR, 'BORDER': 1, 'SLIDER_DEPTH': 0, 'PROGRESS_DEPTH': 0, 'COLOR_LIST': ['#396362', '#4b8e8d', '#f6c89f', '#ffe7d1']}, 'LightBrown7': {'BACKGROUND': '#f6c89f', 'TEXT': '#396362', 'INPUT': '#4b8e8d', 'TEXT_INPUT': '#FFFFFF', 'SCROLL': '#396362', 'BUTTON': ('#FFFFFF', '#396362'), 'PROGRESS': DEFAULT_PROGRESS_BAR_COLOR, 'BORDER': 1, 'SLIDER_DEPTH': 0, 'PROGRESS_DEPTH': 0, 'COLOR_LIST': ['#396362', '#4b8e8d', '#f6c89f', '#ffe7d1']}, 'DarkPurple1': {'BACKGROUND': '#0c093c', 'TEXT': '#fad6d6', 'INPUT': '#eea5f6', 'TEXT_INPUT': '#000000', 'SCROLL': '#eea5f6', 'BUTTON': ('#FFFFFF', '#df42d1'), 'PROGRESS': DEFAULT_PROGRESS_BAR_COLOR, 'BORDER': 1, 'SLIDER_DEPTH': 0, 'PROGRESS_DEPTH': 0, 'COLOR_LIST': ['#0c093c', '#df42d1', '#eea5f6', '#fad6d6']}, 'DarkGrey3': {'BACKGROUND': '#211717', 'TEXT': '#dfddc7', 'INPUT': '#f58b54', 'TEXT_INPUT': '#000000', 'SCROLL': '#f58b54', 'BUTTON': ('#dfddc7', '#a34a28'), 'PROGRESS': DEFAULT_PROGRESS_BAR_COLOR, 'BORDER': 1, 'SLIDER_DEPTH': 0, 'PROGRESS_DEPTH': 0, 'COLOR_LIST': ['#211717', '#a34a28', '#f58b54', '#dfddc7']}, 'LightBrown8': {'BACKGROUND': '#dfddc7', 'TEXT': '#211717', 'INPUT': '#a34a28', 'TEXT_INPUT': '#dfddc7', 'SCROLL': '#211717', 'BUTTON': ('#dfddc7', '#a34a28'), 'PROGRESS': DEFAULT_PROGRESS_BAR_COLOR, 'BORDER': 1, 'SLIDER_DEPTH': 0, 'PROGRESS_DEPTH': 0, 'COLOR_LIST': ['#211717', '#a34a28', '#f58b54', '#dfddc7']}, 'DarkBlue4': {'BACKGROUND': '#494ca2', 'TEXT': '#e3e7f1', 'INPUT': '#c6cbef', 'TEXT_INPUT': '#000000', 'SCROLL': '#c6cbef', 'BUTTON': ('#FFFFFF', '#8186d5'), 'PROGRESS': DEFAULT_PROGRESS_BAR_COLOR, 'BORDER': 1, 'SLIDER_DEPTH': 0, 'PROGRESS_DEPTH': 0, 'COLOR_LIST': ['#494ca2', '#8186d5', '#c6cbef', '#e3e7f1']}, 'LightBlue4': {'BACKGROUND': '#5c94bd', 'TEXT': '#470938', 'INPUT': '#1a3e59', 'TEXT_INPUT': '#FFFFFF', 'SCROLL': '#470938', 'BUTTON': ('#FFFFFF', '#470938'), 'PROGRESS': DEFAULT_PROGRESS_BAR_COLOR, 'BORDER': 1, 'SLIDER_DEPTH': 0, 'PROGRESS_DEPTH': 0, 'COLOR_LIST': ['#470938', '#1a3e59', '#5c94bd', '#f2d6eb']}, 'DarkTeal2': {'BACKGROUND': '#394a6d', 'TEXT': '#c0ffb3', 'INPUT': '#52de97', 'TEXT_INPUT': '#000000', 'SCROLL': '#52de97', 'BUTTON': ('#c0ffb3', '#394a6d'), 'PROGRESS': DEFAULT_PROGRESS_BAR_COLOR, 'BORDER': 1, 'SLIDER_DEPTH': 0, 'PROGRESS_DEPTH': 0, 'COLOR_LIST': ['#394a6d', '#3c9d9b', '#52de97', '#c0ffb3']}, 'DarkTeal3': {'BACKGROUND': '#3c9d9b', 'TEXT': '#c0ffb3', 'INPUT': '#52de97', 'TEXT_INPUT': '#000000', 'SCROLL': '#52de97', 'BUTTON': ('#c0ffb3', '#394a6d'), 'PROGRESS': DEFAULT_PROGRESS_BAR_COLOR, 'BORDER': 1, 'SLIDER_DEPTH': 0, 'PROGRESS_DEPTH': 0, 'COLOR_LIST': ['#394a6d', '#3c9d9b', '#52de97', '#c0ffb3']}, 'DarkPurple5': {'BACKGROUND': '#730068', 'TEXT': '#f6f078', 'INPUT': '#01d28e', 'TEXT_INPUT': '#000000', 'SCROLL': '#01d28e', 'BUTTON': ('#f6f078', '#730068'), 'PROGRESS': DEFAULT_PROGRESS_BAR_COLOR, 'BORDER': 1, 'SLIDER_DEPTH': 0, 'PROGRESS_DEPTH': 0, 'COLOR_LIST': ['#730068', '#434982', '#01d28e', '#f6f078']}, 'DarkPurple2': {'BACKGROUND': '#202060', 'TEXT': '#b030b0', 'INPUT': '#602080', 'TEXT_INPUT': '#FFFFFF', 'SCROLL': '#602080', 'BUTTON': ('#FFFFFF', '#202040'), 'PROGRESS': DEFAULT_PROGRESS_BAR_COLOR, 'BORDER': 1, 'SLIDER_DEPTH': 0, 'PROGRESS_DEPTH': 0, 'COLOR_LIST': ['#202040', '#202060', '#602080', '#b030b0']}, 'DarkBlue5': {'BACKGROUND': '#000272', 'TEXT': '#ff6363', 'INPUT': '#a32f80', 'TEXT_INPUT': '#FFFFFF', 'SCROLL': '#a32f80', 'BUTTON': ('#FFFFFF', '#341677'), 'PROGRESS': DEFAULT_PROGRESS_BAR_COLOR, 'BORDER': 1, 'SLIDER_DEPTH': 0, 'PROGRESS_DEPTH': 0, 'COLOR_LIST': ['#000272', '#341677', '#a32f80', '#ff6363']}, 'LightGrey2': {'BACKGROUND': '#f6f6f6', 'TEXT': '#420000', 'INPUT': '#d4d7dd', 'TEXT_INPUT': '#420000', 'SCROLL': '#420000', 'BUTTON': ('#420000', '#d4d7dd'), 'PROGRESS': DEFAULT_PROGRESS_BAR_COLOR, 'BORDER': 1, 'SLIDER_DEPTH': 0, 'PROGRESS_DEPTH': 0, 'COLOR_LIST': ['#420000', '#d4d7dd', '#eae9e9', '#f6f6f6']}, 'LightGrey3': {'BACKGROUND': '#eae9e9', 'TEXT': '#420000', 'INPUT': '#d4d7dd', 'TEXT_INPUT': '#420000', 'SCROLL': '#420000', 'BUTTON': ('#420000', '#d4d7dd'), 'PROGRESS': DEFAULT_PROGRESS_BAR_COLOR, 'BORDER': 1, 'SLIDER_DEPTH': 0, 'PROGRESS_DEPTH': 0, 'COLOR_LIST': ['#420000', '#d4d7dd', '#eae9e9', '#f6f6f6']}, 'DarkBlue6': {'BACKGROUND': '#01024e', 'TEXT': '#ff6464', 'INPUT': '#8b4367', 'TEXT_INPUT': '#FFFFFF', 'SCROLL': '#8b4367', 'BUTTON': ('#FFFFFF', '#543864'), 'PROGRESS': DEFAULT_PROGRESS_BAR_COLOR, 'BORDER': 1, 'SLIDER_DEPTH': 0, 'PROGRESS_DEPTH': 0, 'COLOR_LIST': ['#01024e', '#543864', '#8b4367', '#ff6464']}, 'DarkBlue7': {'BACKGROUND': '#241663', 'TEXT': '#eae7af', 'INPUT': '#a72693', 'TEXT_INPUT': '#eae7af', 'SCROLL': '#a72693', 'BUTTON': ('#eae7af', '#160f30'), 'PROGRESS': DEFAULT_PROGRESS_BAR_COLOR, 'BORDER': 1, 'SLIDER_DEPTH': 0, 'PROGRESS_DEPTH': 0, 'COLOR_LIST': ['#160f30', '#241663', '#a72693', '#eae7af']}, 'LightBrown9': {'BACKGROUND': '#f6d365', 'TEXT': '#3a1f5d', 'INPUT': '#c83660', 'TEXT_INPUT': '#f6d365', 'SCROLL': '#3a1f5d', 'BUTTON': ('#f6d365', '#c83660'), 'PROGRESS': DEFAULT_PROGRESS_BAR_COLOR, 'BORDER': 1, 'SLIDER_DEPTH': 0, 'PROGRESS_DEPTH': 0, 'COLOR_LIST': ['#3a1f5d', '#c83660', '#e15249', '#f6d365']}, 'DarkPurple3': {'BACKGROUND': '#6e2142', 'TEXT': '#ffd692', 'INPUT': '#e16363', 'TEXT_INPUT': '#ffd692', 'SCROLL': '#e16363', 'BUTTON': ('#ffd692', '#943855'), 'PROGRESS': DEFAULT_PROGRESS_BAR_COLOR, 'BORDER': 1, 'SLIDER_DEPTH': 0, 'PROGRESS_DEPTH': 0, 'COLOR_LIST': ['#6e2142', '#943855', '#e16363', '#ffd692']}, 'LightBrown10': {'BACKGROUND': '#ffd692', 'TEXT': '#6e2142', 'INPUT': '#943855', 'TEXT_INPUT': '#FFFFFF', 'SCROLL': '#6e2142', 'BUTTON': ('#FFFFFF', '#6e2142'), 'PROGRESS': DEFAULT_PROGRESS_BAR_COLOR, 'BORDER': 1, 'SLIDER_DEPTH': 0, 'PROGRESS_DEPTH': 0, 'COLOR_LIST': ['#6e2142', '#943855', '#e16363', '#ffd692']}, 'DarkPurple4': {'BACKGROUND': '#200f21', 'TEXT': '#f638dc', 'INPUT': '#5a3d5c', 'TEXT_INPUT': '#FFFFFF', 'SCROLL': '#5a3d5c', 'BUTTON': ('#FFFFFF', '#382039'), 'PROGRESS': DEFAULT_PROGRESS_BAR_COLOR, 'BORDER': 1, 'SLIDER_DEPTH': 0, 'PROGRESS_DEPTH': 0, 'COLOR_LIST': ['#200f21', '#382039', '#5a3d5c', '#f638dc']}, 'LightBlue5': {'BACKGROUND': '#b2fcff', 'TEXT': '#3e64ff', 'INPUT': '#5edfff', 'TEXT_INPUT': '#000000', 'SCROLL': '#3e64ff', 'BUTTON': ('#FFFFFF', '#3e64ff'), 'PROGRESS': DEFAULT_PROGRESS_BAR_COLOR, 'BORDER': 1, 'SLIDER_DEPTH': 0, 'PROGRESS_DEPTH': 0, 'COLOR_LIST': ['#3e64ff', '#5edfff', '#b2fcff', '#ecfcff']}, 'DarkTeal4': {'BACKGROUND': '#464159', 'TEXT': '#c7f0db', 'INPUT': '#8bbabb', 'TEXT_INPUT': '#000000', 'SCROLL': '#8bbabb', 'BUTTON': ('#FFFFFF', '#6c7b95'), 'PROGRESS': DEFAULT_PROGRESS_BAR_COLOR, 'BORDER': 1, 'SLIDER_DEPTH': 0, 'PROGRESS_DEPTH': 0, 'COLOR_LIST': ['#464159', '#6c7b95', '#8bbabb', '#c7f0db']}, 'LightTeal': {'BACKGROUND': '#c7f0db', 'TEXT': '#464159', 'INPUT': '#6c7b95', 'TEXT_INPUT': '#FFFFFF', 'SCROLL': '#464159', 'BUTTON': ('#FFFFFF', '#464159'), 'PROGRESS': DEFAULT_PROGRESS_BAR_COLOR, 'BORDER': 1, 'SLIDER_DEPTH': 0, 'PROGRESS_DEPTH': 0, 'COLOR_LIST': ['#464159', '#6c7b95', '#8bbabb', '#c7f0db']}, 'DarkTeal5': {'BACKGROUND': '#8bbabb', 'TEXT': '#464159', 'INPUT': '#6c7b95', 'TEXT_INPUT': '#FFFFFF', 'SCROLL': '#464159', 'BUTTON': ('#c7f0db', '#6c7b95'), 'PROGRESS': DEFAULT_PROGRESS_BAR_COLOR, 'BORDER': 1, 'SLIDER_DEPTH': 0, 'PROGRESS_DEPTH': 0, 'COLOR_LIST': ['#464159', '#6c7b95', '#8bbabb', '#c7f0db']}, 'LightGrey4': {'BACKGROUND': '#faf5ef', 'TEXT': '#672f2f', 'INPUT': '#99b19c', 'TEXT_INPUT': '#672f2f', 'SCROLL': '#672f2f', 'BUTTON': ('#672f2f', '#99b19c'), 'PROGRESS': DEFAULT_PROGRESS_BAR_COLOR, 'BORDER': 1, 'SLIDER_DEPTH': 0, 'PROGRESS_DEPTH': 0, 'COLOR_LIST': ['#672f2f', '#99b19c', '#d7d1c9', '#faf5ef']}, 'LightGreen7': {'BACKGROUND': '#99b19c', 'TEXT': '#faf5ef', 'INPUT': '#d7d1c9', 'TEXT_INPUT': '#000000', 'SCROLL': '#d7d1c9', 'BUTTON': ('#FFFFFF', '#99b19c'), 'PROGRESS': DEFAULT_PROGRESS_BAR_COLOR, 'BORDER': 1, 'SLIDER_DEPTH': 0, 'PROGRESS_DEPTH': 0, 'COLOR_LIST': ['#672f2f', '#99b19c', '#d7d1c9', '#faf5ef']}, 'LightGrey5': {'BACKGROUND': '#d7d1c9', 'TEXT': '#672f2f', 'INPUT': '#99b19c', 'TEXT_INPUT': '#672f2f', 'SCROLL': '#672f2f', 'BUTTON': ('#FFFFFF', '#672f2f'), 'PROGRESS': DEFAULT_PROGRESS_BAR_COLOR, 'BORDER': 1, 'SLIDER_DEPTH': 0, 'PROGRESS_DEPTH': 0, 'COLOR_LIST': ['#672f2f', '#99b19c', '#d7d1c9', '#faf5ef']}, 'DarkBrown3': {'BACKGROUND': '#a0855b', 'TEXT': '#f9f6f2', 'INPUT': '#f1d6ab', 'TEXT_INPUT': '#000000', 'SCROLL': '#f1d6ab', 'BUTTON': ('#FFFFFF', '#38470b'), 'PROGRESS': DEFAULT_PROGRESS_BAR_COLOR, 'BORDER': 1, 'SLIDER_DEPTH': 0, 'PROGRESS_DEPTH': 0, 'COLOR_LIST': ['#38470b', '#a0855b', '#f1d6ab', '#f9f6f2']}, 'LightBrown11': {'BACKGROUND': '#f1d6ab', 'TEXT': '#38470b', 'INPUT': '#a0855b', 'TEXT_INPUT': '#FFFFFF', 'SCROLL': '#38470b', 'BUTTON': ('#f9f6f2', '#a0855b'), 'PROGRESS': DEFAULT_PROGRESS_BAR_COLOR, 'BORDER': 1, 'SLIDER_DEPTH': 0, 'PROGRESS_DEPTH': 0, 'COLOR_LIST': ['#38470b', '#a0855b', '#f1d6ab', '#f9f6f2']}, 'DarkRed': {'BACKGROUND': '#83142c', 'TEXT': '#f9d276', 'INPUT': '#ad1d45', 'TEXT_INPUT': '#FFFFFF', 'SCROLL': '#ad1d45', 'BUTTON': ('#f9d276', '#ad1d45'), 'PROGRESS': DEFAULT_PROGRESS_BAR_COLOR, 'BORDER': 1, 'SLIDER_DEPTH': 0, 'PROGRESS_DEPTH': 0, 'COLOR_LIST': ['#44000d', '#83142c', '#ad1d45', '#f9d276']}, 'DarkTeal6': {'BACKGROUND': '#204969', 'TEXT': '#fff7f7', 'INPUT': '#dadada', 'TEXT_INPUT': '#000000', 'SCROLL': '#dadada', 'BUTTON': ('#000000', '#fff7f7'), 'PROGRESS': DEFAULT_PROGRESS_BAR_COLOR, 'BORDER': 1, 'SLIDER_DEPTH': 0, 'PROGRESS_DEPTH': 0, 'COLOR_LIST': ['#204969', '#08ffc8', '#dadada', '#fff7f7']}, 'DarkBrown4': {'BACKGROUND': '#252525', 'TEXT': '#ff0000', 'INPUT': '#af0404', 'TEXT_INPUT': '#FFFFFF', 'SCROLL': '#af0404', 'BUTTON': ('#FFFFFF', '#252525'), 'PROGRESS': DEFAULT_PROGRESS_BAR_COLOR, 'BORDER': 1, 'SLIDER_DEPTH': 0, 'PROGRESS_DEPTH': 0, 'COLOR_LIST': ['#252525', '#414141', '#af0404', '#ff0000']}, 'LightYellow': {'BACKGROUND': '#f4ff61', 'TEXT': '#27aa80', 'INPUT': '#32ff6a', 'TEXT_INPUT': '#000000', 'SCROLL': '#27aa80', 'BUTTON': ('#f4ff61', '#27aa80'), 'PROGRESS': DEFAULT_PROGRESS_BAR_COLOR, 'BORDER': 1, 'SLIDER_DEPTH': 0, 'PROGRESS_DEPTH': 0, 'COLOR_LIST': ['#27aa80', '#32ff6a', '#a8ff3e', '#f4ff61']}, 'DarkGreen1': {'BACKGROUND': '#2b580c', 'TEXT': '#fdef96', 'INPUT': '#f7b71d', 'TEXT_INPUT': '#000000', 'SCROLL': '#f7b71d', 'BUTTON': ('#fdef96', '#2b580c'), 'PROGRESS': DEFAULT_PROGRESS_BAR_COLOR, 'BORDER': 1, 'SLIDER_DEPTH': 0, 'PROGRESS_DEPTH': 0, 'COLOR_LIST': ['#2b580c', '#afa939', '#f7b71d', '#fdef96']}, 'LightGreen8': {'BACKGROUND': '#c8dad3', 'TEXT': '#63707e', 'INPUT': '#93b5b3', 'TEXT_INPUT': '#000000', 'SCROLL': '#63707e', 'BUTTON': ('#FFFFFF', '#63707e'), 'PROGRESS': DEFAULT_PROGRESS_BAR_COLOR, 'BORDER': 1, 'SLIDER_DEPTH': 0, 'PROGRESS_DEPTH': 0, 'COLOR_LIST': ['#63707e', '#93b5b3', '#c8dad3', '#f2f6f5']}, 'DarkTeal7': {'BACKGROUND': '#248ea9', 'TEXT': '#fafdcb', 'INPUT': '#aee7e8', 'TEXT_INPUT': '#000000', 'SCROLL': '#aee7e8', 'BUTTON': ('#000000', '#fafdcb'), 'PROGRESS': DEFAULT_PROGRESS_BAR_COLOR, 'BORDER': 1, 'SLIDER_DEPTH': 0, 'PROGRESS_DEPTH': 0, 'COLOR_LIST': ['#248ea9', '#28c3d4', '#aee7e8', '#fafdcb']}, 'DarkBlue8': {'BACKGROUND': '#454d66', 'TEXT': '#d9d872', 'INPUT': '#58b368', 'TEXT_INPUT': '#000000', 'SCROLL': '#58b368', 'BUTTON': ('#000000', '#009975'), 'PROGRESS': DEFAULT_PROGRESS_BAR_COLOR, 'BORDER': 1, 'SLIDER_DEPTH': 0, 'PROGRESS_DEPTH': 0, 'COLOR_LIST': ['#009975', '#454d66', '#58b368', '#d9d872']}, 'DarkBlue9': {'BACKGROUND': '#263859', 'TEXT': '#ff6768', 'INPUT': '#6b778d', 'TEXT_INPUT': '#FFFFFF', 'SCROLL': '#6b778d', 'BUTTON': ('#ff6768', '#263859'), 'PROGRESS': DEFAULT_PROGRESS_BAR_COLOR, 'BORDER': 1, 'SLIDER_DEPTH': 0, 'PROGRESS_DEPTH': 0, 'COLOR_LIST': ['#17223b', '#263859', '#6b778d', '#ff6768']}, 'DarkBlue10': {'BACKGROUND': '#0028ff', 'TEXT': '#f1f4df', 'INPUT': '#10eaf0', 'TEXT_INPUT': '#000000', 'SCROLL': '#10eaf0', 'BUTTON': ('#f1f4df', '#24009c'), 'PROGRESS': DEFAULT_PROGRESS_BAR_COLOR, 'BORDER': 1, 'SLIDER_DEPTH': 0, 'PROGRESS_DEPTH': 0, 'COLOR_LIST': ['#24009c', '#0028ff', '#10eaf0', '#f1f4df']}, 'DarkBlue11': {'BACKGROUND': '#6384b3', 'TEXT': '#e6f0b6', 'INPUT': '#b8e9c0', 'TEXT_INPUT': '#000000', 'SCROLL': '#b8e9c0', 'BUTTON': ('#e6f0b6', '#684949'), 'PROGRESS': DEFAULT_PROGRESS_BAR_COLOR, 'BORDER': 1, 'SLIDER_DEPTH': 0, 'PROGRESS_DEPTH': 0, 'COLOR_LIST': ['#684949', '#6384b3', '#b8e9c0', '#e6f0b6']}, 'DarkTeal8': {'BACKGROUND': '#71a0a5', 'TEXT': '#212121', 'INPUT': '#665c84', 'TEXT_INPUT': '#FFFFFF', 'SCROLL': '#212121', 'BUTTON': ('#fab95b', '#665c84'), 'PROGRESS': DEFAULT_PROGRESS_BAR_COLOR, 'BORDER': 1, 'SLIDER_DEPTH': 0, 'PROGRESS_DEPTH': 0, 'COLOR_LIST': ['#212121', '#665c84', '#71a0a5', '#fab95b']}, 'DarkRed1': {'BACKGROUND': '#c10000', 'TEXT': '#eeeeee', 'INPUT': '#dedede', 'TEXT_INPUT': '#000000', 'SCROLL': '#dedede', 'BUTTON': ('#c10000', '#eeeeee'), 'PROGRESS': DEFAULT_PROGRESS_BAR_COLOR, 'BORDER': 1, 'SLIDER_DEPTH': 0, 'PROGRESS_DEPTH': 0, 'COLOR_LIST': ['#c10000', '#ff4949', '#dedede', '#eeeeee']}, 'LightBrown5': {'BACKGROUND': '#fff591', 'TEXT': '#e41749', 'INPUT': '#f5587b', 'TEXT_INPUT': '#000000', 'SCROLL': '#e41749', 'BUTTON': ('#fff591', '#e41749'), 'PROGRESS': DEFAULT_PROGRESS_BAR_COLOR, 'BORDER': 1, 'SLIDER_DEPTH': 0, 'PROGRESS_DEPTH': 0, 'COLOR_LIST': ['#e41749', '#f5587b', '#ff8a5c', '#fff591']}, 'LightGreen9': {'BACKGROUND': '#f1edb3', 'TEXT': '#3b503d', 'INPUT': '#4a746e', 'TEXT_INPUT': '#f1edb3', 'SCROLL': '#3b503d', 'BUTTON': ('#f1edb3', '#3b503d'), 'PROGRESS': DEFAULT_PROGRESS_BAR_COLOR, 'BORDER': 1, 'SLIDER_DEPTH': 0, 'PROGRESS_DEPTH': 0, 'COLOR_LIST': ['#3b503d', '#4a746e', '#c8cf94', '#f1edb3'], 'DESCRIPTION': ['Green', 'Turquoise', 'Yellow']}, 'DarkGreen2': {'BACKGROUND': '#3b503d', 'TEXT': '#f1edb3', 'INPUT': '#c8cf94', 'TEXT_INPUT': '#000000', 'SCROLL': '#c8cf94', 'BUTTON': ('#f1edb3', '#3b503d'), 'PROGRESS': DEFAULT_PROGRESS_BAR_COLOR, 'BORDER': 1, 'SLIDER_DEPTH': 0, 'PROGRESS_DEPTH': 0, 'COLOR_LIST': ['#3b503d', '#4a746e', '#c8cf94', '#f1edb3'], 'DESCRIPTION': ['Green', 'Turquoise', 'Yellow']}, 'LightGray1': {'BACKGROUND': '#f2f2f2', 'TEXT': '#222831', 'INPUT': '#393e46', 'TEXT_INPUT': '#FFFFFF', 'SCROLL': '#222831', 'BUTTON': ('#f2f2f2', '#222831'), 'PROGRESS': DEFAULT_PROGRESS_BAR_COLOR, 'BORDER': 1, 'SLIDER_DEPTH': 0, 'PROGRESS_DEPTH': 0, 'COLOR_LIST': ['#222831', '#393e46', '#f96d00', '#f2f2f2'], 'DESCRIPTION': ['#000000', 'Grey', 'Orange', 'Grey', 'Autumn']}, 'DarkGrey4': {'BACKGROUND': '#52524e', 'TEXT': '#e9e9e5', 'INPUT': '#d4d6c8', 'TEXT_INPUT': '#000000', 'SCROLL': '#d4d6c8', 'BUTTON': ('#FFFFFF', '#9a9b94'), 'PROGRESS': DEFAULT_PROGRESS_BAR_COLOR, 'BORDER': 1, 'SLIDER_DEPTH': 0, 'PROGRESS_DEPTH': 0, 'COLOR_LIST': ['#52524e', '#9a9b94', '#d4d6c8', '#e9e9e5'], 'DESCRIPTION': ['Grey', 'Pastel', 'Winter']}, 'DarkBlue12': {'BACKGROUND': '#324e7b', 'TEXT': '#f8f8f8', 'INPUT': '#86a6df', 'TEXT_INPUT': '#000000', 'SCROLL': '#86a6df', 'BUTTON': ('#FFFFFF', '#5068a9'), 'PROGRESS': DEFAULT_PROGRESS_BAR_COLOR, 'BORDER': 1, 'SLIDER_DEPTH': 0, 'PROGRESS_DEPTH': 0, 'COLOR_LIST': ['#324e7b', '#5068a9', '#86a6df', '#f8f8f8'], 'DESCRIPTION': ['Blue', 'Grey', 'Cold', 'Winter']}, 'DarkPurple6': {'BACKGROUND': '#070739', 'TEXT': '#e1e099', 'INPUT': '#c327ab', 'TEXT_INPUT': '#e1e099', 'SCROLL': '#c327ab', 'BUTTON': ('#e1e099', '#521477'), 'PROGRESS': DEFAULT_PROGRESS_BAR_COLOR, 'BORDER': 1, 'SLIDER_DEPTH': 0, 'PROGRESS_DEPTH': 0, 'COLOR_LIST': ['#070739', '#521477', '#c327ab', '#e1e099'], 'DESCRIPTION': ['#000000', 'Purple', 'Yellow', 'Dark']}, 'DarkBlue13': {'BACKGROUND': '#203562', 'TEXT': '#e3e8f8', 'INPUT': '#c0c5cd', 'TEXT_INPUT': '#000000', 'SCROLL': '#c0c5cd', 'BUTTON': ('#FFFFFF', '#3e588f'), 'PROGRESS': DEFAULT_PROGRESS_BAR_COLOR, 'BORDER': 1, 'SLIDER_DEPTH': 0, 'PROGRESS_DEPTH': 0, 'COLOR_LIST': ['#203562', '#3e588f', '#c0c5cd', '#e3e8f8'], 'DESCRIPTION': ['Blue', 'Grey', 'Wedding', 'Cold']}, 'DarkBrown5': {'BACKGROUND': '#3c1b1f', 'TEXT': '#f6e1b5', 'INPUT': '#e2bf81', 'TEXT_INPUT': '#000000', 'SCROLL': '#e2bf81', 'BUTTON': ('#3c1b1f', '#f6e1b5'), 'PROGRESS': DEFAULT_PROGRESS_BAR_COLOR, 'BORDER': 1, 'SLIDER_DEPTH': 0, 'PROGRESS_DEPTH': 0, 'COLOR_LIST': ['#3c1b1f', '#b21e4b', '#e2bf81', '#f6e1b5'], 'DESCRIPTION': ['Brown', 'Red', 'Yellow', 'Warm']}, 'DarkGreen3': {'BACKGROUND': '#062121', 'TEXT': '#eeeeee', 'INPUT': '#e4dcad', 'TEXT_INPUT': '#000000', 'SCROLL': '#e4dcad', 'BUTTON': ('#eeeeee', '#181810'), 'PROGRESS': DEFAULT_PROGRESS_BAR_COLOR, 'BORDER': 1, 'SLIDER_DEPTH': 0, 'PROGRESS_DEPTH': 0, 'COLOR_LIST': ['#062121', '#181810', '#e4dcad', '#eeeeee'], 'DESCRIPTION': ['#000000', '#000000', 'Brown', 'Grey']}, 'DarkBlack1': {'BACKGROUND': '#181810', 'TEXT': '#eeeeee', 'INPUT': '#e4dcad', 'TEXT_INPUT': '#000000', 'SCROLL': '#e4dcad', 'BUTTON': ('#FFFFFF', '#062121'), 'PROGRESS': DEFAULT_PROGRESS_BAR_COLOR, 'BORDER': 1, 'SLIDER_DEPTH': 0, 'PROGRESS_DEPTH': 0, 'COLOR_LIST': ['#062121', '#181810', '#e4dcad', '#eeeeee'], 'DESCRIPTION': ['#000000', '#000000', 'Brown', 'Grey']}, 'DarkGrey5': {'BACKGROUND': '#343434', 'TEXT': '#f3f3f3', 'INPUT': '#e9dcbe', 'TEXT_INPUT': '#000000', 'SCROLL': '#e9dcbe', 'BUTTON': ('#FFFFFF', '#8e8b82'), 'PROGRESS': DEFAULT_PROGRESS_BAR_COLOR, 'BORDER': 1, 'SLIDER_DEPTH': 0, 'PROGRESS_DEPTH': 0, 'COLOR_LIST': ['#343434', '#8e8b82', '#e9dcbe', '#f3f3f3'], 'DESCRIPTION': ['Grey', 'Brown']}, 'LightBrown12': {'BACKGROUND': '#8e8b82', 'TEXT': '#f3f3f3', 'INPUT': '#e9dcbe', 'TEXT_INPUT': '#000000', 'SCROLL': '#e9dcbe', 'BUTTON': ('#f3f3f3', '#8e8b82'), 'PROGRESS': DEFAULT_PROGRESS_BAR_COLOR, 'BORDER': 1, 'SLIDER_DEPTH': 0, 'PROGRESS_DEPTH': 0, 'COLOR_LIST': ['#343434', '#8e8b82', '#e9dcbe', '#f3f3f3'], 'DESCRIPTION': ['Grey', 'Brown']}, 'DarkTeal9': {'BACKGROUND': '#13445a', 'TEXT': '#fef4e8', 'INPUT': '#446878', 'TEXT_INPUT': '#FFFFFF', 'SCROLL': '#446878', 'BUTTON': ('#fef4e8', '#446878'), 'PROGRESS': DEFAULT_PROGRESS_BAR_COLOR, 'BORDER': 1, 'SLIDER_DEPTH': 0, 'PROGRESS_DEPTH': 0, 'COLOR_LIST': ['#13445a', '#970747', '#446878', '#fef4e8'], 'DESCRIPTION': ['Red', 'Grey', 'Blue', 'Wedding', 'Retro']}, 'DarkBlue14': {'BACKGROUND': '#21273d', 'TEXT': '#f1f6f8', 'INPUT': '#b9d4f1', 'TEXT_INPUT': '#000000', 'SCROLL': '#b9d4f1', 'BUTTON': ('#FFFFFF', '#6a759b'), 'PROGRESS': DEFAULT_PROGRESS_BAR_COLOR, 'BORDER': 1, 'SLIDER_DEPTH': 0, 'PROGRESS_DEPTH': 0, 'COLOR_LIST': ['#21273d', '#6a759b', '#b9d4f1', '#f1f6f8'], 'DESCRIPTION': ['Blue', '#000000', 'Grey', 'Cold', 'Winter']}, 'LightBlue6': {'BACKGROUND': '#f1f6f8', 'TEXT': '#21273d', 'INPUT': '#6a759b', 'TEXT_INPUT': '#FFFFFF', 'SCROLL': '#21273d', 'BUTTON': ('#f1f6f8', '#6a759b'), 'PROGRESS': DEFAULT_PROGRESS_BAR_COLOR, 'BORDER': 1, 'SLIDER_DEPTH': 0, 'PROGRESS_DEPTH': 0, 'COLOR_LIST': ['#21273d', '#6a759b', '#b9d4f1', '#f1f6f8'], 'DESCRIPTION': ['Blue', '#000000', 'Grey', 'Cold', 'Winter']}, 'DarkGreen4': {'BACKGROUND': '#044343', 'TEXT': '#e4e4e4', 'INPUT': '#045757', 'TEXT_INPUT': '#e4e4e4', 'SCROLL': '#045757', 'BUTTON': ('#e4e4e4', '#045757'), 'PROGRESS': DEFAULT_PROGRESS_BAR_COLOR, 'BORDER': 1, 'SLIDER_DEPTH': 0, 'PROGRESS_DEPTH': 0, 'COLOR_LIST': ['#222222', '#044343', '#045757', '#e4e4e4'], 'DESCRIPTION': ['#000000', 'Turquoise', 'Grey', 'Dark']}, 'DarkGreen5': {'BACKGROUND': '#1b4b36', 'TEXT': '#e0e7f1', 'INPUT': '#aebd77', 'TEXT_INPUT': '#000000', 'SCROLL': '#aebd77', 'BUTTON': ('#FFFFFF', '#538f6a'), 'PROGRESS': DEFAULT_PROGRESS_BAR_COLOR, 'BORDER': 1, 'SLIDER_DEPTH': 0, 'PROGRESS_DEPTH': 0, 'COLOR_LIST': ['#1b4b36', '#538f6a', '#aebd77', '#e0e7f1'], 'DESCRIPTION': ['Green', 'Grey']}, 'DarkTeal10': {'BACKGROUND': '#0d3446', 'TEXT': '#d8dfe2', 'INPUT': '#71adb5', 'TEXT_INPUT': '#000000', 'SCROLL': '#71adb5', 'BUTTON': ('#FFFFFF', '#176d81'), 'PROGRESS': DEFAULT_PROGRESS_BAR_COLOR, 'BORDER': 1, 'SLIDER_DEPTH': 0, 'PROGRESS_DEPTH': 0, 'COLOR_LIST': ['#0d3446', '#176d81', '#71adb5', '#d8dfe2'], 'DESCRIPTION': ['Grey', 'Turquoise', 'Winter', 'Cold']}, 'DarkGrey6': {'BACKGROUND': '#3e3e3e', 'TEXT': '#ededed', 'INPUT': '#68868c', 'TEXT_INPUT': '#ededed', 'SCROLL': '#68868c', 'BUTTON': ('#FFFFFF', '#405559'), 'PROGRESS': DEFAULT_PROGRESS_BAR_COLOR, 'BORDER': 1, 'SLIDER_DEPTH': 0, 'PROGRESS_DEPTH': 0, 'COLOR_LIST': ['#3e3e3e', '#405559', '#68868c', '#ededed'], 'DESCRIPTION': ['Grey', 'Turquoise', 'Winter']}, 'DarkTeal11': {'BACKGROUND': '#405559', 'TEXT': '#ededed', 'INPUT': '#68868c', 'TEXT_INPUT': '#ededed', 'SCROLL': '#68868c', 'BUTTON': ('#ededed', '#68868c'), 'PROGRESS': DEFAULT_PROGRESS_BAR_COLOR, 'BORDER': 1, 'SLIDER_DEPTH': 0, 'PROGRESS_DEPTH': 0, 'COLOR_LIST': ['#3e3e3e', '#405559', '#68868c', '#ededed'], 'DESCRIPTION': ['Grey', 'Turquoise', 'Winter']}, 'LightBlue7': {'BACKGROUND': '#9ed0e0', 'TEXT': '#19483f', 'INPUT': '#5c868e', 'TEXT_INPUT': '#FFFFFF', 'SCROLL': '#19483f', 'BUTTON': ('#FFFFFF', '#19483f'), 'PROGRESS': DEFAULT_PROGRESS_BAR_COLOR, 'BORDER': 1, 'SLIDER_DEPTH': 0, 'PROGRESS_DEPTH': 0, 'COLOR_LIST': ['#19483f', '#5c868e', '#ff6a38', '#9ed0e0'], 'DESCRIPTION': ['Orange', 'Blue', 'Turquoise']}, 'LightGreen10': {'BACKGROUND': '#d8ebb5', 'TEXT': '#205d67', 'INPUT': '#639a67', 'TEXT_INPUT': '#FFFFFF', 'SCROLL': '#205d67', 'BUTTON': ('#d8ebb5', '#205d67'), 'PROGRESS': DEFAULT_PROGRESS_BAR_COLOR, 'BORDER': 1, 'SLIDER_DEPTH': 0, 'PROGRESS_DEPTH': 0, 'COLOR_LIST': ['#205d67', '#639a67', '#d9bf77', '#d8ebb5'], 'DESCRIPTION': ['Blue', 'Green', 'Brown', 'Vintage']}, 'DarkBlue15': {'BACKGROUND': '#151680', 'TEXT': '#f1fea4', 'INPUT': '#375fc0', 'TEXT_INPUT': '#f1fea4', 'SCROLL': '#375fc0', 'BUTTON': ('#f1fea4', '#1c44ac'), 'PROGRESS': DEFAULT_PROGRESS_BAR_COLOR, 'BORDER': 1, 'SLIDER_DEPTH': 0, 'PROGRESS_DEPTH': 0, 'COLOR_LIST': ['#151680', '#1c44ac', '#375fc0', '#f1fea4'], 'DESCRIPTION': ['Blue', 'Yellow', 'Cold']}, 'DarkBlue16': {'BACKGROUND': '#1c44ac', 'TEXT': '#f1fea4', 'INPUT': '#375fc0', 'TEXT_INPUT': '#f1fea4', 'SCROLL': '#375fc0', 'BUTTON': ('#f1fea4', '#151680'), 'PROGRESS': DEFAULT_PROGRESS_BAR_COLOR, 'BORDER': 1, 'SLIDER_DEPTH': 0, 'PROGRESS_DEPTH': 0, 'COLOR_LIST': ['#151680', '#1c44ac', '#375fc0', '#f1fea4'], 'DESCRIPTION': ['Blue', 'Yellow', 'Cold']}, 'DarkTeal12': {'BACKGROUND': '#004a7c', 'TEXT': '#fafafa', 'INPUT': '#e8f1f5', 'TEXT_INPUT': '#000000', 'SCROLL': '#e8f1f5', 'BUTTON': ('#fafafa', '#005691'), 'PROGRESS': DEFAULT_PROGRESS_BAR_COLOR, 'BORDER': 1, 'SLIDER_DEPTH': 0, 'PROGRESS_DEPTH': 0, 'COLOR_LIST': ['#004a7c', '#005691', '#e8f1f5', '#fafafa'], 'DESCRIPTION': ['Grey', 'Blue', 'Cold', 'Winter']}, 'LightBrown13': {'BACKGROUND': '#ebf5ee', 'TEXT': '#921224', 'INPUT': '#bdc6b8', 'TEXT_INPUT': '#921224', 'SCROLL': '#921224', 'BUTTON': ('#FFFFFF', '#921224'), 'PROGRESS': DEFAULT_PROGRESS_BAR_COLOR, 'BORDER': 1, 'SLIDER_DEPTH': 0, 'PROGRESS_DEPTH': 0, 'COLOR_LIST': ['#921224', '#bdc6b8', '#bce0da', '#ebf5ee'], 'DESCRIPTION': ['Red', 'Blue', 'Grey', 'Vintage', 'Wedding']}, 'DarkBlue17': {'BACKGROUND': '#21294c', 'TEXT': '#f9f2d7', 'INPUT': '#f2dea8', 'TEXT_INPUT': '#000000', 'SCROLL': '#f2dea8', 'BUTTON': ('#f9f2d7', '#141829'), 'PROGRESS': DEFAULT_PROGRESS_BAR_COLOR, 'BORDER': 1, 'SLIDER_DEPTH': 0, 'PROGRESS_DEPTH': 0, 'COLOR_LIST': ['#141829', '#21294c', '#f2dea8', '#f9f2d7'], 'DESCRIPTION': ['#000000', 'Blue', 'Yellow']}, 'DarkBrown6': {'BACKGROUND': '#785e4d', 'TEXT': '#f2eee3', 'INPUT': '#baaf92', 'TEXT_INPUT': '#000000', 'SCROLL': '#baaf92', 'BUTTON': ('#FFFFFF', '#785e4d'), 'PROGRESS': DEFAULT_PROGRESS_BAR_COLOR, 'BORDER': 1, 'SLIDER_DEPTH': 0, 'PROGRESS_DEPTH': 0, 'COLOR_LIST': ['#785e4d', '#ff8426', '#baaf92', '#f2eee3'], 'DESCRIPTION': ['Grey', 'Brown', 'Orange', 'Autumn']}, 'DarkGreen6': {'BACKGROUND': '#5c715e', 'TEXT': '#f2f9f1', 'INPUT': '#ddeedf', 'TEXT_INPUT': '#000000', 'SCROLL': '#ddeedf', 'BUTTON': ('#f2f9f1', '#5c715e'), 'PROGRESS': DEFAULT_PROGRESS_BAR_COLOR, 'BORDER': 1, 'SLIDER_DEPTH': 0, 'PROGRESS_DEPTH': 0, 'COLOR_LIST': ['#5c715e', '#b6cdbd', '#ddeedf', '#f2f9f1'], 'DESCRIPTION': ['Grey', 'Green', 'Vintage']}, 'DarkGrey7': {'BACKGROUND': '#4b586e', 'TEXT': '#dddddd', 'INPUT': '#574e6d', 'TEXT_INPUT': '#dddddd', 'SCROLL': '#574e6d', 'BUTTON': ('#dddddd', '#43405d'), 'PROGRESS': DEFAULT_PROGRESS_BAR_COLOR, 'BORDER': 1, 'SLIDER_DEPTH': 0, 'PROGRESS_DEPTH': 0, 'COLOR_LIST': ['#43405d', '#4b586e', '#574e6d', '#dddddd'], 'DESCRIPTION': ['Grey', 'Winter', 'Cold']}, 'DarkRed2': {'BACKGROUND': '#ab1212', 'TEXT': '#f6e4b5', 'INPUT': '#cd3131', 'TEXT_INPUT': '#f6e4b5', 'SCROLL': '#cd3131', 'BUTTON': ('#f6e4b5', '#ab1212'), 'PROGRESS': DEFAULT_PROGRESS_BAR_COLOR, 'BORDER': 1, 'SLIDER_DEPTH': 0, 'PROGRESS_DEPTH': 0, 'COLOR_LIST': ['#ab1212', '#1fad9f', '#cd3131', '#f6e4b5'], 'DESCRIPTION': ['Turquoise', 'Red', 'Yellow']}, 'LightGrey6': {'BACKGROUND': '#e3e3e3', 'TEXT': '#233142', 'INPUT': '#455d7a', 'TEXT_INPUT': '#e3e3e3', 'SCROLL': '#233142', 'BUTTON': ('#e3e3e3', '#455d7a'), 'PROGRESS': DEFAULT_PROGRESS_BAR_COLOR, 'BORDER': 1, 'SLIDER_DEPTH': 0, 'PROGRESS_DEPTH': 0, 'COLOR_LIST': ['#233142', '#455d7a', '#f95959', '#e3e3e3'], 'DESCRIPTION': ['#000000', 'Blue', 'Red', 'Grey']}, 'HotDogStand': {'BACKGROUND': 'red', 'TEXT': 'yellow', 'INPUT': 'yellow', 'TEXT_INPUT': '#000000', 'SCROLL': 'yellow', 'BUTTON': ('red', 'yellow'), 'PROGRESS': DEFAULT_PROGRESS_BAR_COLOR, 'BORDER': 1, 'SLIDER_DEPTH': 0, 'PROGRESS_DEPTH': 0}}

def ListOfLookAndFeelValues():
    if False:
        while True:
            i = 10
    '\n    Get a list of the valid values to pass into your call to change_look_and_feel\n    :return: List[str] - list of valid string values\n    '
    return sorted(list(LOOK_AND_FEEL_TABLE.keys()))

def theme(new_theme=None):
    if False:
        return 10
    '\n    Sets / Gets the current Theme.  If none is specified then returns the current theme.\n    This call replaces the ChangeLookAndFeel / change_look_and_feel call which only sets the theme.\n\n    :param new_theme: (str) the new theme name to use\n    :return: (str) the currently selected theme\n    '
    if new_theme is not None:
        change_look_and_feel(new_theme)
    return CURRENT_LOOK_AND_FEEL

def theme_background_color(color=None):
    if False:
        while True:
            i = 10
    '\n    Sets/Returns the background color currently in use\n    Used for Windows and containers (Column, Frame, Tab) and tables\n\n    :return: (str) - color string of the background color currently in use\n    '
    if color is not None:
        set_options(background_color=color)
    return DEFAULT_BACKGROUND_COLOR

def theme_element_background_color(color=None):
    if False:
        for i in range(10):
            print('nop')
    '\n    Sets/Returns the background color currently in use for all elements except containers\n\n    :return: (str) - color string of the element background color currently in use\n    '
    if color is not None:
        set_options(element_background_color=color)
    return DEFAULT_ELEMENT_BACKGROUND_COLOR

def theme_text_color(color=None):
    if False:
        return 10
    '\n    Sets/Returns the text color currently in use\n\n    :return: (str) - color string of the text color currently in use\n    '
    if color is not None:
        set_options(text_color=color)
    return DEFAULT_TEXT_COLOR

def theme_text_element_background_color(color=None):
    if False:
        i = 10
        return i + 15
    '\n    Sets/Returns the background color for text elements\n\n    :return: (str) - color string of the text background color currently in use\n    '
    if color is not None:
        set_options(text_element_background_color=color)
    return DEFAULT_TEXT_ELEMENT_BACKGROUND_COLOR

def theme_input_background_color(color=None):
    if False:
        for i in range(10):
            print('nop')
    '\n    Sets/Returns the input element background color currently in use\n\n    :return: (str) - color string of the input element background color currently in use\n    '
    if color is not None:
        set_options(input_elements_background_color=color)
    return DEFAULT_INPUT_ELEMENTS_COLOR

def theme_input_text_color(color=None):
    if False:
        return 10
    "\n    Sets/Returns the input element entry color (not the text but the thing that's displaying the text)\n\n    :return: (str) - color string of the input element color currently in use\n    "
    if color is not None:
        set_options(input_text_color=color)
    return DEFAULT_INPUT_TEXT_COLOR

def theme_button_color(color=None):
    if False:
        for i in range(10):
            print('nop')
    '\n    Sets/Returns the button color currently in use\n\n    :return: Tuple[str, str] - TUPLE with color strings of the button color currently in use (button text color, button background color)\n    '
    if color is not None:
        set_options(button_color=color)
    return DEFAULT_BUTTON_COLOR

def theme_progress_bar_color(color=None):
    if False:
        print('Hello World!')
    '\n    Sets/Returns the progress bar colors by the current color theme\n\n    :return: Tuple[str, str] - TUPLE with color strings of the ProgressBar color currently in use(button text color, button background color)\n    '
    if color is not None:
        set_options(progress_meter_color=color)
    return DEFAULT_PROGRESS_BAR_COLOR

def theme_slider_color(color=None):
    if False:
        for i in range(10):
            print('nop')
    '\n    Sets/Returns the slider color (used for sliders)\n\n    :return: (str) - color string of the slider color currently in use\n    '
    if color is not None:
        set_options(scrollbar_color=color)
    return DEFAULT_SCROLLBAR_COLOR

def theme_border_width(border_width=None):
    if False:
        print('Hello World!')
    '\n    Sets/Returns the border width currently in use\n    Used by non ttk elements at the moment\n\n    :return: (int) - border width currently in use\n    '
    if border_width is not None:
        set_options(border_width=border_width)
    return DEFAULT_BORDER_WIDTH

def theme_slider_border_width(border_width=None):
    if False:
        return 10
    '\n    Sets/Returns the slider border width currently in use\n\n    :return: (int) - border width currently in use\n    '
    if border_width is not None:
        set_options(slider_border_width=border_width)
    return DEFAULT_SLIDER_BORDER_WIDTH

def theme_progress_bar_border_width(border_width=None):
    if False:
        print('Hello World!')
    '\n    Sets/Returns the progress meter border width currently in use\n\n    :return: (int) - border width currently in use\n    '
    if border_width is not None:
        set_options(progress_meter_border_depth=border_width)
    return DEFAULT_PROGRESS_BAR_BORDER_WIDTH

def theme_element_text_color(color=None):
    if False:
        for i in range(10):
            print('nop')
    '\n    Sets/Returns the text color used by elements that have text as part of their display (Tables, Trees and Sliders)\n\n    :return: (str) - color string currently in use\n    '
    if color is not None:
        set_options(element_text_color=color)
    return DEFAULT_ELEMENT_TEXT_COLOR

def theme_list():
    if False:
        return 10
    '\n    Returns a sorted list of the currently available color themes\n\n    :return: List[str] - A sorted list of the currently available color themes\n    '
    return list_of_look_and_feel_values()

def theme_add_new(new_theme_name, new_theme_dict):
    if False:
        for i in range(10):
            print('nop')
    '\n    Add a new theme to the dictionary of themes\n\n    :param new_theme_name: text to display in element\n    :type new_theme_name: (str)\n    :param new_theme_dict: text to display in element\n    :type new_theme_dict: (dict)\n    '
    global LOOK_AND_FEEL_TABLE
    try:
        LOOK_AND_FEEL_TABLE[new_theme_name] = new_theme_dict
    except Exception as e:
        print('Exception during adding new theme {}'.format(e))

def theme_previewer(columns=12):
    if False:
        print('Hello World!')
    '\n    Show a window with all of the color themes - takes a while so be patient\n\n    :param columns: (int) number of themes in a single row\n    '
    preview_all_look_and_feel_themes(columns)

def ChangeLookAndFeel(index, force=False):
    if False:
        i = 10
        return i + 15
    '\n    Change the "color scheme" of all future PySimpleGUI Windows.\n    The scheme are string names that specify a group of colors. Background colors, text colors, button colors.\n    There are 13 different color settings that are changed at one time using a single call to ChangeLookAndFeel\n    The look and feel table itself has these indexes into the dictionary LOOK_AND_FEEL_TABLE.\n    The original list was (prior to a major rework and renaming)... these names still work...\n    In Nov 2019 a new Theme Formula was devised to make choosing a theme easier:\n    The "Formula" is:\n    ["Dark" or "Light"] Color Number\n    Colors can be Blue Brown Grey Green Purple Red Teal Yellow Black\n    The number will vary for each pair. There are more DarkGrey entries than there are LightYellow for example.\n    Default = The default settings (only button color is different than system default)\n    Default1 = The full system default including the button (everything\'s gray... how sad... don\'t be all gray... please....)\n    :param index: (str) the name of the index into the Look and Feel table (does not have to be exact, can be "fuzzy")\n    :param force: (bool) no longer used\n    '
    global CURRENT_LOOK_AND_FEEL
    theme = index
    lf_values = [item.lower() for item in list_of_look_and_feel_values()]
    opt1 = theme.replace(' ', '').lower()
    optx = theme.lower().split(' ')
    optx.reverse()
    opt2 = ''.join(optx)
    if opt1 in lf_values:
        ix = lf_values.index(opt1)
    elif opt2 in lf_values:
        ix = lf_values.index(opt2)
    else:
        ix = randint(0, len(lf_values) - 1)
        print('** Warning - {} Theme is not a valid theme. Change your theme call. **'.format(index))
        print('valid values are', list_of_look_and_feel_values())
        print('Instead, please enjoy a random Theme named {}'.format(list_of_look_and_feel_values()[ix]))
    selection = list_of_look_and_feel_values()[ix]
    CURRENT_LOOK_AND_FEEL = selection
    try:
        colors = LOOK_AND_FEEL_TABLE[selection]
        if colors['PROGRESS'] != COLOR_SYSTEM_DEFAULT:
            if colors['BUTTON'][1] != colors['INPUT'] and colors['BUTTON'][1] != colors['BACKGROUND']:
                colors['PROGRESS'] = (colors['BUTTON'][1], colors['INPUT'])
            else:
                colors['PROGRESS'] = (colors['TEXT_INPUT'], colors['INPUT'])
        else:
            colors['PROGRESS'] = DEFAULT_PROGRESS_BAR_COLOR_OFFICIAL
        SetOptions(background_color=colors['BACKGROUND'], text_element_background_color=colors['BACKGROUND'], element_background_color=colors['BACKGROUND'], text_color=colors['TEXT'], input_elements_background_color=colors['INPUT'], button_color=colors['BUTTON'], progress_meter_color=colors['PROGRESS'], border_width=colors['BORDER'], slider_border_width=colors['SLIDER_DEPTH'], progress_meter_border_depth=colors['PROGRESS_DEPTH'], scrollbar_color=colors['SCROLL'], element_text_color=colors['TEXT'], input_text_color=colors['TEXT_INPUT'])
    except:
        print('** Warning - Theme value not valid. Change your theme call. **')
        print('valid values are', list_of_look_and_feel_values())

def preview_all_look_and_feel_themes(columns=12):
    if False:
        for i in range(10):
            print('nop')
    '\n    Displays a "Quick Reference Window" showing all of the different Look and Feel settings that are available.\n    They are sorted alphabetically.  The legacy color names are mixed in, but otherwise they are sorted into Dark and Light halves\n    :param columns: (int) The number of themes to display per row\n    '
    popup_quick_message('Hang on for a moment, this will take a bit to create....', background_color='red', text_color='#FFFFFF', auto_close=True, non_blocking=True)
    web = False
    win_bg = 'black'

    def sample_layout():
        if False:
            for i in range(10):
                print('nop')
        return [[Text('Text element'), InputText('Input data here', size=(10, 1))], [Button('Ok'), Button('Cancel'), Slider((1, 10), orientation='h', size=(5, 15))]]
    layout = [[Text('Here is a complete list of themes', font='Default 18', background_color=win_bg)]]
    names = list_of_look_and_feel_values()
    names.sort()
    row = []
    for (count, theme) in enumerate(names):
        change_look_and_feel(theme)
        if not count % columns:
            layout += [row]
            row = []
        row += [Frame(theme, sample_layout() if not web else [[T(theme)]] + sample_layout())]
    if row:
        layout += [row]
    window = Window('Preview of all Look and Feel choices', layout, background_color=win_bg)
    window.read()
    window.close()
sprint = ScrolledTextBox

def ObjToStringSingleObj(obj):
    if False:
        for i in range(10):
            print('nop')
    if obj is None:
        return 'None'
    return str(obj.__class__) + '\n' + '\n'.join((repr(item) + ' = ' + repr(obj.__dict__[item]) for item in sorted(obj.__dict__)))

def ObjToString(obj, extra='    '):
    if False:
        while True:
            i = 10
    if obj is None:
        return 'None'
    return str(obj.__class__) + '\n' + '\n'.join((extra + (str(item) + ' = ' + (ObjToString(obj.__dict__[item], extra + '    ') if hasattr(obj.__dict__[item], '__dict__') else str(obj.__dict__[item]))) for item in sorted(obj.__dict__)))

def Popup(*args, button_color=None, background_color=None, text_color=None, button_type=POPUP_BUTTONS_OK, auto_close=False, auto_close_duration=None, custom_text=(None, None), non_blocking=False, icon=DEFAULT_WINDOW_ICON, line_width=None, font=None, no_titlebar=False, grab_anywhere=False, keep_on_top=False, location=(None, None)):
    if False:
        while True:
            i = 10
    '\n    Popup - Display a popup box with as many parms as you wish to include\n    :param args:\n    :param button_color:\n    :param background_color:\n    :param text_color:\n    :param button_type:\n    :param auto_close:\n    :param auto_close_duration:\n    :param non_blocking:\n    :param icon:\n    :param line_width:\n    :param font:\n    :param no_titlebar:\n    :param grab_anywhere:\n    :param keep_on_top:\n    :param location:\n    :return:\n    '
    if not args:
        args_to_print = ['']
    else:
        args_to_print = args
    if line_width != None:
        local_line_width = line_width
    else:
        local_line_width = MESSAGE_BOX_LINE_WIDTH
    title = args_to_print[0] if args_to_print[0] is not None else 'None'
    window = Window(title, auto_size_text=True, background_color=background_color, button_color=button_color, auto_close=auto_close, auto_close_duration=auto_close_duration, icon=icon, font=font, no_titlebar=no_titlebar, grab_anywhere=grab_anywhere, keep_on_top=keep_on_top, location=location)
    (max_line_total, total_lines) = (0, 0)
    for message in args_to_print:
        message = str(message)
        if message.count('\n'):
            message_wrapped = message
        else:
            message_wrapped = textwrap.fill(message, local_line_width)
        message_wrapped_lines = message_wrapped.count('\n') + 1
        longest_line_len = max([len(l) for l in message.split('\n')])
        width_used = min(longest_line_len, local_line_width)
        max_line_total = max(max_line_total, width_used)
        height = message_wrapped_lines
        window.AddRow(Text(message_wrapped, auto_size_text=True, text_color=text_color, background_color=background_color))
        total_lines += height
    if non_blocking:
        PopupButton = DummyButton
    else:
        PopupButton = Button
    if custom_text != (None, None):
        if type(custom_text) is not tuple:
            window.AddRow(PopupButton(custom_text, size=(len(custom_text), 1), button_color=button_color, focus=True, bind_return_key=True))
        elif custom_text[1] is None:
            window.AddRow(PopupButton(custom_text[0], size=(len(custom_text[0]), 1), button_color=button_color, focus=True, bind_return_key=True))
        else:
            window.AddRow(PopupButton(custom_text[0], button_color=button_color, focus=True, bind_return_key=True, size=(len(custom_text[0]), 1)), PopupButton(custom_text[1], button_color=button_color, size=(len(custom_text[0]), 1)))
    elif button_type is POPUP_BUTTONS_YES_NO:
        window.AddRow(PopupButton('Yes', button_color=button_color, focus=True, bind_return_key=True, pad=((20, 5), 3), size=(5, 1)), PopupButton('No', button_color=button_color, size=(5, 1)))
    elif button_type is POPUP_BUTTONS_CANCELLED:
        window.AddRow(PopupButton('Cancelled', button_color=button_color, focus=True, bind_return_key=True, pad=((20, 0), 3)))
    elif button_type is POPUP_BUTTONS_ERROR:
        window.AddRow(PopupButton('Error', size=(6, 1), button_color=button_color, focus=True, bind_return_key=True, pad=((20, 0), 3)))
    elif button_type is POPUP_BUTTONS_OK_CANCEL:
        window.AddRow(PopupButton('OK', size=(6, 1), button_color=button_color, focus=True, bind_return_key=True), PopupButton('Cancel', size=(6, 1), button_color=button_color))
    elif button_type is POPUP_BUTTONS_NO_BUTTONS:
        pass
    else:
        window.AddRow(PopupButton('OK', size=(5, 1), button_color=button_color, focus=True, bind_return_key=True, pad=((20, 0), 3)))
    if non_blocking:
        (button, values) = window.Read(timeout=0)
    else:
        (button, values) = window.Read()
        window.Close()
    return button

def MsgBox(*args):
    if False:
        for i in range(10):
            print('nop')
    raise DeprecationWarning('MsgBox is no longer supported... change your call to Popup')

def PopupNoButtons(*args, button_color=None, background_color=None, text_color=None, auto_close=False, auto_close_duration=None, non_blocking=False, icon=DEFAULT_WINDOW_ICON, line_width=None, font=None, no_titlebar=False, grab_anywhere=False, keep_on_top=False, location=(None, None)):
    if False:
        return 10
    '\n    Show a Popup but without any buttons\n    :param args:\n    :param button_color:\n    :param background_color:\n    :param text_color:\n    :param auto_close:\n    :param auto_close_duration:\n    :param non_blocking:\n    :param icon:\n    :param line_width:\n    :param font:\n    :param no_titlebar:\n    :param grab_anywhere:\n    :param keep_on_top:\n    :param location:\n    :return:\n    '
    Popup(*args, button_color=button_color, background_color=background_color, text_color=text_color, button_type=POPUP_BUTTONS_NO_BUTTONS, auto_close=auto_close, auto_close_duration=auto_close_duration, non_blocking=non_blocking, icon=icon, line_width=line_width, font=font, no_titlebar=no_titlebar, grab_anywhere=grab_anywhere, keep_on_top=keep_on_top, location=location)

def PopupNonBlocking(*args, button_type=POPUP_BUTTONS_OK, button_color=None, background_color=None, text_color=None, auto_close=False, auto_close_duration=None, non_blocking=True, icon=DEFAULT_WINDOW_ICON, line_width=None, font=None, no_titlebar=False, grab_anywhere=False, keep_on_top=False, location=(None, None)):
    if False:
        return 10
    '\n    Show Popup box and immediately return (does not block)\n    :param args:\n    :param button_type:\n    :param button_color:\n    :param background_color:\n    :param text_color:\n    :param auto_close:\n    :param auto_close_duration:\n    :param non_blocking:\n    :param icon:\n    :param line_width:\n    :param font:\n    :param no_titlebar:\n    :param grab_anywhere:\n    :param keep_on_top:\n    :param location:\n    :return:\n    '
    Popup(*args, button_color=button_color, background_color=background_color, text_color=text_color, button_type=button_type, auto_close=auto_close, auto_close_duration=auto_close_duration, non_blocking=non_blocking, icon=icon, line_width=line_width, font=font, no_titlebar=no_titlebar, grab_anywhere=grab_anywhere, keep_on_top=keep_on_top, location=location)
PopupNoWait = PopupNonBlocking

def PopupQuick(*args, button_type=POPUP_BUTTONS_OK, button_color=None, background_color=None, text_color=None, auto_close=True, auto_close_duration=2, non_blocking=True, icon=DEFAULT_WINDOW_ICON, line_width=None, font=None, no_titlebar=False, grab_anywhere=False, keep_on_top=False, location=(None, None)):
    if False:
        while True:
            i = 10
    "\n    Show Popup box that doesn't block and closes itself\n    :param args:\n    :param button_type:\n    :param button_color:\n    :param background_color:\n    :param text_color:\n    :param auto_close:\n    :param auto_close_duration:\n    :param non_blocking:\n    :param icon:\n    :param line_width:\n    :param font:\n    :param no_titlebar:\n    :param grab_anywhere:\n    :param keep_on_top:\n    :param location:\n    :return:\n    "
    Popup(*args, button_color=button_color, background_color=background_color, text_color=text_color, button_type=button_type, auto_close=auto_close, auto_close_duration=auto_close_duration, non_blocking=non_blocking, icon=icon, line_width=line_width, font=font, no_titlebar=no_titlebar, grab_anywhere=grab_anywhere, keep_on_top=keep_on_top, location=location)

def PopupQuickMessage(*args, button_type=POPUP_BUTTONS_NO_BUTTONS, button_color=None, background_color=None, text_color=None, auto_close=True, auto_close_duration=2, non_blocking=True, icon=DEFAULT_WINDOW_ICON, line_width=None, font=None, no_titlebar=True, grab_anywhere=False, keep_on_top=False, location=(None, None)):
    if False:
        while True:
            i = 10
    "\n    Show Popup box that doesn't block and closes itself\n    :param args:\n    :param button_type:\n    :param button_color:\n    :param background_color:\n    :param text_color:\n    :param auto_close:\n    :param auto_close_duration:\n    :param non_blocking:\n    :param icon:\n    :param line_width:\n    :param font:\n    :param no_titlebar:\n    :param grab_anywhere:\n    :param keep_on_top:\n    :param location:\n    :return:\n    "
    Popup(*args, button_color=button_color, background_color=background_color, text_color=text_color, button_type=button_type, auto_close=auto_close, auto_close_duration=auto_close_duration, non_blocking=non_blocking, icon=icon, line_width=line_width, font=font, no_titlebar=no_titlebar, grab_anywhere=grab_anywhere, keep_on_top=keep_on_top, location=location)

def PopupNoTitlebar(*args, button_type=POPUP_BUTTONS_OK, button_color=None, background_color=None, text_color=None, auto_close=False, auto_close_duration=None, non_blocking=False, icon=DEFAULT_WINDOW_ICON, line_width=None, font=None, grab_anywhere=True, keep_on_top=False, location=(None, None)):
    if False:
        i = 10
        return i + 15
    '\n    Display a Popup without a titlebar.   Enables grab anywhere so you can move it\n    :param args:\n    :param button_type:\n    :param button_color:\n    :param background_color:\n    :param text_color:\n    :param auto_close:\n    :param auto_close_duration:\n    :param non_blocking:\n    :param icon:\n    :param line_width:\n    :param font:\n    :param grab_anywhere:\n    :param keep_on_top:\n    :param location:\n    :return:\n    '
    Popup(*args, button_color=button_color, background_color=background_color, text_color=text_color, button_type=button_type, auto_close=auto_close, auto_close_duration=auto_close_duration, non_blocking=non_blocking, icon=icon, line_width=line_width, font=font, no_titlebar=True, grab_anywhere=grab_anywhere, keep_on_top=keep_on_top, location=location)
PopupNoFrame = PopupNoTitlebar
PopupNoBorder = PopupNoTitlebar
PopupAnnoying = PopupNoTitlebar

def PopupAutoClose(*args, button_type=POPUP_BUTTONS_OK, button_color=None, background_color=None, text_color=None, auto_close=True, auto_close_duration=None, non_blocking=False, icon=DEFAULT_WINDOW_ICON, line_width=None, font=None, no_titlebar=False, grab_anywhere=False, keep_on_top=False, location=(None, None)):
    if False:
        while True:
            i = 10
    '\n    Popup that closes itself after some time period\n    :param args:\n    :param button_type:\n    :param button_color:\n    :param background_color:\n    :param text_color:\n    :param auto_close:\n    :param auto_close_duration:\n    :param non_blocking:\n    :param icon:\n    :param line_width:\n    :param font:\n    :param no_titlebar:\n    :param grab_anywhere:\n    :param keep_on_top:\n    :param location:\n    :return:\n    '
    Popup(*args, button_color=button_color, background_color=background_color, text_color=text_color, button_type=button_type, auto_close=auto_close, auto_close_duration=auto_close_duration, non_blocking=non_blocking, icon=icon, line_width=line_width, font=font, no_titlebar=no_titlebar, grab_anywhere=grab_anywhere, keep_on_top=keep_on_top, location=location)
PopupTimed = PopupAutoClose

def PopupError(*args, button_color=DEFAULT_ERROR_BUTTON_COLOR, background_color=None, text_color=None, auto_close=False, auto_close_duration=None, non_blocking=False, icon=DEFAULT_WINDOW_ICON, line_width=None, font=None, no_titlebar=False, grab_anywhere=False, keep_on_top=False, location=(None, None)):
    if False:
        i = 10
        return i + 15
    "\n    Popup with colored button and 'Error' as button text\n    :param args:\n    :param button_color:\n    :param background_color:\n    :param text_color:\n    :param auto_close:\n    :param auto_close_duration:\n    :param non_blocking:\n    :param icon:\n    :param line_width:\n    :param font:\n    :param no_titlebar:\n    :param grab_anywhere:\n    :param keep_on_top:\n    :param location:\n    :return:\n    "
    Popup(*args, button_type=POPUP_BUTTONS_ERROR, background_color=background_color, text_color=text_color, non_blocking=non_blocking, icon=icon, line_width=line_width, button_color=button_color, auto_close=auto_close, auto_close_duration=auto_close_duration, font=font, no_titlebar=no_titlebar, grab_anywhere=grab_anywhere, keep_on_top=keep_on_top, location=location)

def PopupCancel(*args, button_color=None, background_color=None, text_color=None, auto_close=False, auto_close_duration=None, non_blocking=False, icon=DEFAULT_WINDOW_ICON, line_width=None, font=None, no_titlebar=False, grab_anywhere=False, keep_on_top=False, location=(None, None)):
    if False:
        while True:
            i = 10
    '\n    Display Popup with "cancelled" button text\n    :param args:\n    :param button_color:\n    :param background_color:\n    :param text_color:\n    :param auto_close:\n    :param auto_close_duration:\n    :param non_blocking:\n    :param icon:\n    :param line_width:\n    :param font:\n    :param no_titlebar:\n    :param grab_anywhere:\n    :param keep_on_top:\n    :param location:\n    :return:\n    '
    Popup(*args, button_type=POPUP_BUTTONS_CANCELLED, background_color=background_color, text_color=text_color, non_blocking=non_blocking, icon=icon, line_width=line_width, button_color=button_color, auto_close=auto_close, auto_close_duration=auto_close_duration, font=font, no_titlebar=no_titlebar, grab_anywhere=grab_anywhere, keep_on_top=keep_on_top, location=location)

def PopupOK(*args, button_color=None, background_color=None, text_color=None, auto_close=False, auto_close_duration=None, non_blocking=False, icon=DEFAULT_WINDOW_ICON, line_width=None, font=None, no_titlebar=False, grab_anywhere=False, keep_on_top=False, location=(None, None)):
    if False:
        i = 10
        return i + 15
    '\n    Display Popup with OK button only\n    :param args:\n    :param button_color:\n    :param background_color:\n    :param text_color:\n    :param auto_close:\n    :param auto_close_duration:\n    :param non_blocking:\n    :param icon:\n    :param line_width:\n    :param font:\n    :param no_titlebar:\n    :param grab_anywhere:\n    :param keep_on_top:\n    :param location:\n    :return:\n    '
    Popup(*args, button_type=POPUP_BUTTONS_OK, background_color=background_color, text_color=text_color, non_blocking=non_blocking, icon=icon, line_width=line_width, button_color=button_color, auto_close=auto_close, auto_close_duration=auto_close_duration, font=font, no_titlebar=no_titlebar, grab_anywhere=grab_anywhere, keep_on_top=keep_on_top, location=location)

def PopupOKCancel(*args, button_color=None, background_color=None, text_color=None, auto_close=False, auto_close_duration=None, non_blocking=False, icon=DEFAULT_WINDOW_ICON, line_width=None, font=None, no_titlebar=False, grab_anywhere=False, keep_on_top=False, location=(None, None)):
    if False:
        for i in range(10):
            print('nop')
    '\n    Display popup with OK and Cancel buttons\n    :param args:\n    :param button_color:\n    :param background_color:\n    :param text_color:\n    :param auto_close:\n    :param auto_close_duration:\n    :param non_blocking:\n    :param icon:\n    :param line_width:\n    :param font:\n    :param no_titlebar:\n    :param grab_anywhere:\n    :param keep_on_top:\n    :param location:\n    :return: OK, Cancel or None\n    '
    return Popup(*args, button_type=POPUP_BUTTONS_OK_CANCEL, background_color=background_color, text_color=text_color, non_blocking=non_blocking, icon=icon, line_width=line_width, button_color=button_color, auto_close=auto_close, auto_close_duration=auto_close_duration, font=font, no_titlebar=no_titlebar, grab_anywhere=grab_anywhere, keep_on_top=keep_on_top, location=location)

def PopupYesNo(*args, button_color=None, background_color=None, text_color=None, auto_close=False, auto_close_duration=None, non_blocking=False, icon=DEFAULT_WINDOW_ICON, line_width=None, font=None, no_titlebar=False, grab_anywhere=False, keep_on_top=False, location=(None, None)):
    if False:
        i = 10
        return i + 15
    '\n    Display Popup with Yes and No buttons\n    :param args:\n    :param button_color:\n    :param background_color:\n    :param text_color:\n    :param auto_close:\n    :param auto_close_duration:\n    :param non_blocking:\n    :param icon:\n    :param line_width:\n    :param font:\n    :param no_titlebar:\n    :param grab_anywhere:\n    :param keep_on_top:\n    :param location:\n    :return: Yes, No or None\n    '
    return Popup(*args, button_type=POPUP_BUTTONS_YES_NO, background_color=background_color, text_color=text_color, non_blocking=non_blocking, icon=icon, line_width=line_width, button_color=button_color, auto_close=auto_close, auto_close_duration=auto_close_duration, font=font, no_titlebar=no_titlebar, grab_anywhere=grab_anywhere, keep_on_top=keep_on_top, location=location)

def PopupGetFolder(message, default_path='', no_window=False, size=(None, None), button_color=None, background_color=None, text_color=None, icon=DEFAULT_WINDOW_ICON, font=None, no_titlebar=False, grab_anywhere=False, keep_on_top=False, location=(None, None), initial_folder=None):
    if False:
        while True:
            i = 10
    '\n    Display popup with text entry field and browse button. Browse for folder\n    :param message:\n    :param default_path:\n    :param no_window:\n    :param size:\n    :param button_color:\n    :param background_color:\n    :param text_color:\n    :param icon:\n    :param font:\n    :param no_titlebar:\n    :param grab_anywhere:\n    :param keep_on_top:\n    :param location:\n    :return: Contents of text field. None if closed using X or cancelled\n    '
    global _my_windows
    if no_window:
        if _my_windows._NumOpenWindows:
            root = tk.Toplevel()
        else:
            root = tk.Tk()
        try:
            root.attributes('-alpha', 0)
        except:
            pass
        folder_name = tk.filedialog.askdirectory()
        root.destroy()
        return folder_name
    layout = [[Text(message, auto_size_text=True, text_color=text_color, background_color=background_color)], [InputText(default_text=default_path, size=size, key='_INPUT_'), FolderBrowse(initial_folder=initial_folder)], [Button('Ok', size=(5, 1), bind_return_key=True), Button('Cancel', size=(5, 1))]]
    window = Window(title=message, layout=layout, icon=icon, auto_size_text=True, button_color=button_color, background_color=background_color, font=font, no_titlebar=no_titlebar, grab_anywhere=grab_anywhere, keep_on_top=keep_on_top, location=location)
    (button, values) = window.Read()
    window.Close()
    if button != 'Ok':
        return None
    else:
        path = values['_INPUT_']
        return path

def PopupGetFile(message, default_path='', default_extension='', save_as=False, file_types=(('ALL Files', '*.*'),), no_window=False, size=(None, None), button_color=None, background_color=None, text_color=None, icon=DEFAULT_WINDOW_ICON, font=None, no_titlebar=False, grab_anywhere=False, keep_on_top=False, location=(None, None), initial_folder=None):
    if False:
        for i in range(10):
            print('nop')
    '\n        Display popup with text entry field and browse button. Browse for file\n    :param message:\n    :param default_path:\n    :param default_extension:\n    :param save_as:\n    :param file_types:\n    :param no_window:\n    :param size:\n    :param button_color:\n    :param background_color:\n    :param text_color:\n    :param icon:\n    :param font:\n    :param no_titlebar:\n    :param grab_anywhere:\n    :param keep_on_top:\n    :param location:\n    :return:  string representing the path chosen, None if cancelled or window closed with X\n    '
    global _my_windows
    if no_window:
        if _my_windows._NumOpenWindows:
            root = tk.Toplevel()
        else:
            root = tk.Tk()
        try:
            root.attributes('-alpha', 0)
        except:
            pass
        if save_as:
            filename = tk.filedialog.asksaveasfilename(filetypes=file_types, defaultextension=default_extension)
        else:
            filename = tk.filedialog.askopenfilename(filetypes=file_types, defaultextension=default_extension)
        root.destroy()
        return filename
    browse_button = SaveAs(file_types=file_types, initial_folder=initial_folder) if save_as else FileBrowse(file_types=file_types, initial_folder=initial_folder)
    layout = [[Text(message, auto_size_text=True, text_color=text_color, background_color=background_color)], [InputText(default_text=default_path, size=size, key='_INPUT_'), browse_button], [Button('Ok', size=(6, 1), bind_return_key=True), Button('Cancel', size=(6, 1))]]
    window = Window(title=message, layout=layout, icon=icon, auto_size_text=True, button_color=button_color, font=font, background_color=background_color, no_titlebar=no_titlebar, grab_anywhere=grab_anywhere, keep_on_top=keep_on_top, location=location)
    (button, values) = window.Read()
    window.Close()
    if button != 'Ok':
        return None
    else:
        path = values['_INPUT_']
        return path

def PopupGetText(message, default_text='', password_char='', size=(None, None), button_color=None, background_color=None, text_color=None, icon=DEFAULT_WINDOW_ICON, font=None, no_titlebar=False, grab_anywhere=False, keep_on_top=False, location=(None, None)):
    if False:
        return 10
    '\n    Display Popup with text entry field\n    :param message:\n    :param default_text:\n    :param password_char:\n    :param size:\n    :param button_color:\n    :param background_color:\n    :param text_color:\n    :param icon:\n    :param font:\n    :param no_titlebar:\n    :param grab_anywhere:\n    :param keep_on_top:\n    :param location:\n    :return: Text entered or None if window was closed\n    '
    layout = [[Text(message, auto_size_text=True, text_color=text_color, background_color=background_color, font=font)], [InputText(default_text=default_text, size=size, key='_INPUT_', password_char=password_char)], [Button('Ok', size=(5, 1), bind_return_key=True), Button('Cancel', size=(5, 1))]]
    window = Window(title=message, layout=layout, icon=icon, auto_size_text=True, button_color=button_color, no_titlebar=no_titlebar, background_color=background_color, grab_anywhere=grab_anywhere, keep_on_top=keep_on_top, location=location)
    (button, values) = window.Read()
    window.Close()
    if button != 'Ok':
        return None
    else:
        path = values['_INPUT_']
        return path
change_look_and_feel = ChangeLookAndFeel
easy_print = EasyPrint
easy_print_close = EasyPrintClose
get_complimentary_hex = GetComplimentaryHex
list_of_look_and_feel_values = ListOfLookAndFeelValues
obj_to_string = ObjToString
obj_to_string_single_obj = ObjToStringSingleObj
one_line_progress_meter = OneLineProgressMeter
one_line_progress_meter_cancel = OneLineProgressMeterCancel
popup = Popup
popup_annoying = PopupAnnoying
popup_auto_close = PopupAutoClose
popup_cancel = PopupCancel
popup_error = PopupError
popup_get_file = PopupGetFile
popup_get_folder = PopupGetFolder
popup_get_text = PopupGetText
popup_no_border = PopupNoBorder
popup_no_buttons = PopupNoButtons
popup_no_frame = PopupNoFrame
popup_no_titlebar = PopupNoTitlebar
popup_no_wait = PopupNoWait
popup_non_blocking = PopupNonBlocking
popup_ok = PopupOK
popup_ok_cancel = PopupOKCancel
popup_quick = PopupQuick
popup_quick_message = PopupQuickMessage
popup_scrolled = PopupScrolled
popup_timed = PopupTimed
popup_yes_no = PopupYesNo
print_close = PrintClose
rgb = RGB
scrolled_text_box = ScrolledTextBox
set_global_icon = SetGlobalIcon
set_options = SetOptions
timer_start = TimerStart
timer_stop = TimerStop
sprint = sprint
theme(CURRENT_LOOK_AND_FEEL)

def main():
    if False:
        print('Hello World!')
    ver = version.split('\n')[0]

    def VerLine(version, description, size=(30, 1)):
        if False:
            i = 10
            return i + 15
        return [Column([[T(description, font='Courier 18', text_color='yellow')], [T(version, font='Courier 18', size=size)]])]
    menu_def = [['&File', ['&Open', '&Save', 'E&xit', 'Properties']], ['&Edit', ['Paste', ['Special', 'Normal'], '!Undo']], ['!&Disabled', ['Paste', ['Special', 'Normal'], '!Undo']], ['&Help', '&About...']]
    menu_def = [['File', ['&Open::mykey', '&Save', 'E&xit', 'Properties']], ['Edit', ['!Paste', ['Special', 'Normal'], '!Undo']], ['!Disabled', ['Has Sub', ['Item1', 'Item2'], 'No Sub']], ['Help', 'About...']]
    col1 = [[Text('Column 1 line  1', background_color='red')], [Text('Column 1 line 2')]]
    layout = [[Menu(menu_def, key='_MENU_', text_color='yellow', background_color='#475841', font='Courier 14')], [Text('PySimpleGUIWeb Welcomes You...', tooltip='text', font=('Comic sans ms', 20), size=(40, 1), text_color='yellow', enable_events=False, key='_PySimpleGUIWeb_')], [T('System platform = %s' % sys.platform)], [Image(data=DEFAULT_BASE64_ICON, enable_events=False)], VerLine(ver, 'PySimpleGUI Version'), VerLine(os.path.dirname(os.path.abspath(__file__)), 'PySimpleGUI Location'), VerLine(sys.version, 'Python Version', size=(60, 2)), VerLine(pkg_resources.get_distribution('remi').version, 'Remi Version'), [T('Current Time '), Text('Text', key='_TEXT_', font='Arial 18', text_color='black', size=(30, 1)), Column(col1, background_color='red')], [T('Up Time'), Text('Text', key='_TEXT_UPTIME_', font='Arial 18', text_color='black', size=(30, 1))], [Input('Single Line Input', do_not_clear=True, enable_events=False, size=(30, 1), text_color='red', key='_IN_')], [Multiline('Multiline Input', do_not_clear=True, size=(40, 4), enable_events=False, key='_MULTI_IN_')], [MultilineOutput('Multiline Output', size=(80, 8), text_color='blue', font='Courier 12', key='_MULTIOUT_', autoscroll=True)], [Checkbox('Checkbox 1', enable_events=True, key='_CB1_'), Checkbox('Checkbox 2', default=True, key='_CB2_', enable_events=True)], [Combo(values=['Combo 1', 'Combo 2', 'Combo 3'], default_value='Combo 2', key='_COMBO_', enable_events=True, readonly=False, tooltip='Combo box', disabled=False, size=(12, 1))], [Listbox(values=('Listbox 1', 'Listbox 2', 'Listbox 3'), enable_events=True, size=(10, 3), key='_LIST_')], [Slider((1, 100), default_value=80, key='_SLIDER_', visible=True, enable_events=True, orientation='v')], [Spin(values=(1, 2, 3), initial_value='2', size=(4, 1), key='_SPIN_', enable_events=True)], [OK(), Button('Hidden', visible=False, key='_HIDDEN_'), Button('Values'), Button('Exit', button_color=('white', 'red')), Button('UnHide'), B('Popup')]]
    window = Window('PySimpleGUIWeb Test Harness Window', layout, font='Arial 18', icon=DEFAULT_BASE64_ICON, default_element_size=(12, 1), auto_size_buttons=False)
    start_time = datetime.datetime.now()
    while True:
        (event, values) = window.Read(timeout=100)
        window.Element('_TEXT_').Update(str(datetime.datetime.now()))
        window.Element('_TEXT_UPTIME_').Update(str(datetime.datetime.now() - start_time))
        print(event, values) if event != TIMEOUT_KEY else None
        if event in (None, 'Exit'):
            break
        elif event == 'OK':
            window.Element('_MULTIOUT_').print('You clicked the OK button')
            window.Element('_PySimpleGUIWeb_').Widget.style['background-image'] = "url('/my_resources:mine.png')"
        elif event == 'Values':
            window.Element('_MULTIOUT_').Update(str(values) + '\n', append=True)
        elif event != TIMEOUT_KEY:
            window.Element('_MULTIOUT_').print('EVENT: ' + str(event))
        if event == 'Popup':
            Popup('This is a popup!')
        if event == 'UnHide':
            print('Unhiding...')
            window.Element('_HIDDEN_').Update(visible=True)
    window.Close()
if __name__ == '__main__':
    main()
    exit(0)