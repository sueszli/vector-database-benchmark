import sys
import warnings
import tkinter
import tkinter.ttk as ttk
from typing import Union, Callable, Tuple
try:
    from typing import TypedDict
except ImportError:
    from typing_extensions import TypedDict
from .... import windows
from ..theme import ThemeManager
from ..font import CTkFont
from ..image import CTkImage
from ..appearance_mode import CTkAppearanceModeBaseClass
from ..scaling import CTkScalingBaseClass
from ..utility import pop_from_dict_by_set, check_kwargs_empty

class CTkBaseClass(tkinter.Frame, CTkAppearanceModeBaseClass, CTkScalingBaseClass):
    """ Base class of every CTk widget, handles the dimensions, bg_color,
        appearance_mode changes, scaling, bg changes of master if master is not a CTk widget """
    _valid_tk_frame_attributes: set = {'cursor'}
    _cursor_manipulation_enabled: bool = True

    def __init__(self, master: any, width: int=0, height: int=0, bg_color: Union[str, Tuple[str, str]]='transparent', **kwargs):
        if False:
            return 10
        tkinter.Frame.__init__(self, master=master, width=width, height=height, **pop_from_dict_by_set(kwargs, self._valid_tk_frame_attributes))
        CTkAppearanceModeBaseClass.__init__(self)
        CTkScalingBaseClass.__init__(self, scaling_type='widget')
        check_kwargs_empty(kwargs, raise_error=True)
        self._current_width = width
        self._current_height = height
        self._desired_width = width
        self._desired_height = height
        super().configure(width=self._apply_widget_scaling(self._desired_width), height=self._apply_widget_scaling(self._desired_height))

        class GeometryCallDict(TypedDict):
            function: Callable
            kwargs: dict
        self._last_geometry_manager_call: Union[GeometryCallDict, None] = None
        self._bg_color: Union[str, Tuple[str, str]] = self._detect_color_of_master() if bg_color == 'transparent' else self._check_color_type(bg_color, transparency=True)
        super().configure(bg=self._apply_appearance_mode(self._bg_color))
        super().bind('<Configure>', self._update_dimensions_event)
        if isinstance(self.master, (tkinter.Tk, tkinter.Toplevel, tkinter.Frame, tkinter.LabelFrame, ttk.Frame, ttk.LabelFrame, ttk.Notebook)) and (not isinstance(self.master, (CTkBaseClass, CTkAppearanceModeBaseClass))):
            master_old_configure = self.master.config

            def new_configure(*args, **kwargs):
                if False:
                    print('Hello World!')
                if 'bg' in kwargs:
                    self.configure(bg_color=kwargs['bg'])
                elif 'background' in kwargs:
                    self.configure(bg_color=kwargs['background'])
                elif len(args) > 0 and type(args[0]) == dict:
                    if 'bg' in args[0]:
                        self.configure(bg_color=args[0]['bg'])
                    elif 'background' in args[0]:
                        self.configure(bg_color=args[0]['background'])
                master_old_configure(*args, **kwargs)
            self.master.config = new_configure
            self.master.configure = new_configure

    def destroy(self):
        if False:
            return 10
        ' Destroy this and all descendants widgets. '
        tkinter.Frame.destroy(self)
        CTkAppearanceModeBaseClass.destroy(self)
        CTkScalingBaseClass.destroy(self)

    def _draw(self, no_color_updates: bool=False):
        if False:
            i = 10
            return i + 15
        ' can be overridden but super method must be called '
        if no_color_updates is False:
            pass

    def config(self, *args, **kwargs):
        if False:
            print('Hello World!')
        raise AttributeError("'config' is not implemented for CTk widgets. For consistency, always use 'configure' instead.")

    def configure(self, require_redraw=False, **kwargs):
        if False:
            i = 10
            return i + 15
        ' basic configure with bg_color, width, height support, calls configure of tkinter.Frame, updates in the end '
        if 'width' in kwargs:
            self._set_dimensions(width=kwargs.pop('width'))
        if 'height' in kwargs:
            self._set_dimensions(height=kwargs.pop('height'))
        if 'bg_color' in kwargs:
            new_bg_color = self._check_color_type(kwargs.pop('bg_color'), transparency=True)
            if new_bg_color == 'transparent':
                self._bg_color = self._detect_color_of_master()
            else:
                self._bg_color = self._check_color_type(new_bg_color)
            require_redraw = True
        super().configure(**pop_from_dict_by_set(kwargs, self._valid_tk_frame_attributes))
        check_kwargs_empty(kwargs, raise_error=True)
        if require_redraw:
            self._draw()

    def cget(self, attribute_name: str):
        if False:
            while True:
                i = 10
        ' basic cget with bg_color, width, height support, calls cget of tkinter.Frame '
        if attribute_name == 'bg_color':
            return self._bg_color
        elif attribute_name == 'width':
            return self._desired_width
        elif attribute_name == 'height':
            return self._desired_height
        elif attribute_name in self._valid_tk_frame_attributes:
            return super().cget(attribute_name)
        else:
            raise ValueError(f"'{attribute_name}' is not a supported argument. Look at the documentation for supported arguments.")

    def _check_font_type(self, font: any):
        if False:
            i = 10
            return i + 15
        ' check font type when passed to widget '
        if isinstance(font, CTkFont):
            return font
        elif type(font) == tuple and len(font) == 1:
            warnings.warn(f'{type(self).__name__} Warning: font {font} given without size, will be extended with default text size of current theme\n')
            return (font[0], ThemeManager.theme['text']['size'])
        elif type(font) == tuple and 2 <= len(font) <= 6:
            return font
        else:
            raise ValueError(f'Wrong font type {type(font)}\n' + f'For consistency, Customtkinter requires the font argument to be a tuple of len 2 to 6 or an instance of CTkFont.\n' + f'\nUsage example:\n' + f"font=customtkinter.CTkFont(family='<name>', size=<size in px>)\n" + f"font=('<name>', <size in px>)\n")

    def _check_image_type(self, image: any):
        if False:
            print('Hello World!')
        ' check image type when passed to widget '
        if image is None:
            return image
        elif isinstance(image, CTkImage):
            return image
        else:
            warnings.warn(f'{type(self).__name__} Warning: Given image is not CTkImage but {type(image)}. Image can not be scaled on HighDPI displays, use CTkImage instead.\n')
            return image

    def _update_dimensions_event(self, event):
        if False:
            print('Hello World!')
        if round(self._current_width) != round(self._reverse_widget_scaling(event.width)) or round(self._current_height) != round(self._reverse_widget_scaling(event.height)):
            self._current_width = self._reverse_widget_scaling(event.width)
            self._current_height = self._reverse_widget_scaling(event.height)
            self._draw(no_color_updates=True)

    def _detect_color_of_master(self, master_widget=None) -> Union[str, Tuple[str, str]]:
        if False:
            for i in range(10):
                print('nop')
        ' detect foreground color of master widget for bg_color and transparent color '
        if master_widget is None:
            master_widget = self.master
        if isinstance(master_widget, (windows.widgets.core_widget_classes.CTkBaseClass, windows.CTk, windows.CTkToplevel, windows.widgets.ctk_scrollable_frame.CTkScrollableFrame)):
            if master_widget.cget('fg_color') is not None and master_widget.cget('fg_color') != 'transparent':
                return master_widget.cget('fg_color')
            elif isinstance(master_widget, windows.widgets.ctk_scrollable_frame.CTkScrollableFrame):
                return self._detect_color_of_master(master_widget.master.master.master)
            elif hasattr(master_widget, 'master'):
                return self._detect_color_of_master(master_widget.master)
        elif isinstance(master_widget, (ttk.Frame, ttk.LabelFrame, ttk.Notebook, ttk.Label)):
            try:
                ttk_style = ttk.Style()
                return ttk_style.lookup(master_widget.winfo_class(), 'background')
            except Exception:
                return ('#FFFFFF', '#000000')
        else:
            try:
                return master_widget.cget('bg')
            except Exception:
                return ('#FFFFFF', '#000000')

    def _set_appearance_mode(self, mode_string):
        if False:
            print('Hello World!')
        super()._set_appearance_mode(mode_string)
        self._draw()
        super().update_idletasks()

    def _set_scaling(self, new_widget_scaling, new_window_scaling):
        if False:
            while True:
                i = 10
        super()._set_scaling(new_widget_scaling, new_window_scaling)
        super().configure(width=self._apply_widget_scaling(self._desired_width), height=self._apply_widget_scaling(self._desired_height))
        if self._last_geometry_manager_call is not None:
            self._last_geometry_manager_call['function'](**self._apply_argument_scaling(self._last_geometry_manager_call['kwargs']))

    def _set_dimensions(self, width=None, height=None):
        if False:
            for i in range(10):
                print('nop')
        if width is not None:
            self._desired_width = width
        if height is not None:
            self._desired_height = height
        super().configure(width=self._apply_widget_scaling(self._desired_width), height=self._apply_widget_scaling(self._desired_height))

    def bind(self, sequence=None, command=None, add=None):
        if False:
            for i in range(10):
                print('nop')
        raise NotImplementedError

    def unbind(self, sequence=None, funcid=None):
        if False:
            print('Hello World!')
        raise NotImplementedError

    def unbind_all(self, sequence):
        if False:
            while True:
                i = 10
        raise AttributeError("'unbind_all' is not allowed, because it would delete necessary internal callbacks for all widgets")

    def bind_all(self, sequence=None, func=None, add=None):
        if False:
            for i in range(10):
                print('nop')
        raise AttributeError("'bind_all' is not allowed, could result in undefined behavior")

    def place(self, **kwargs):
        if False:
            i = 10
            return i + 15
        '\n        Place a widget in the parent widget. Use as options:\n        in=master - master relative to which the widget is placed\n        in_=master - see \'in\' option description\n        x=amount - locate anchor of this widget at position x of master\n        y=amount - locate anchor of this widget at position y of master\n        relx=amount - locate anchor of this widget between 0.0 and 1.0 relative to width of master (1.0 is right edge)\n        rely=amount - locate anchor of this widget between 0.0 and 1.0 relative to height of master (1.0 is bottom edge)\n        anchor=NSEW (or subset) - position anchor according to given direction\n        width=amount - width of this widget in pixel\n        height=amount - height of this widget in pixel\n        relwidth=amount - width of this widget between 0.0 and 1.0 relative to width of master (1.0 is the same width as the master)\n        relheight=amount - height of this widget between 0.0 and 1.0 relative to height of master (1.0 is the same height as the master)\n        bordermode="inside" or "outside" - whether to take border width of master widget into account\n        '
        if 'width' in kwargs or 'height' in kwargs:
            raise ValueError("'width' and 'height' arguments must be passed to the constructor of the widget, not the place method")
        self._last_geometry_manager_call = {'function': super().place, 'kwargs': kwargs}
        return super().place(**self._apply_argument_scaling(kwargs))

    def place_forget(self):
        if False:
            for i in range(10):
                print('nop')
        ' Unmap this widget. '
        self._last_geometry_manager_call = None
        return super().place_forget()

    def pack(self, **kwargs):
        if False:
            while True:
                i = 10
        "\n        Pack a widget in the parent widget. Use as options:\n        after=widget - pack it after you have packed widget\n        anchor=NSEW (or subset) - position widget according to given direction\n        before=widget - pack it before you will pack widget\n        expand=bool - expand widget if parent size grows\n        fill=NONE or X or Y or BOTH - fill widget if widget grows\n        in=master - use master to contain this widget\n        in_=master - see 'in' option description\n        ipadx=amount - add internal padding in x direction\n        ipady=amount - add internal padding in y direction\n        padx=amount - add padding in x direction\n        pady=amount - add padding in y direction\n        side=TOP or BOTTOM or LEFT or RIGHT -  where to add this widget.\n        "
        self._last_geometry_manager_call = {'function': super().pack, 'kwargs': kwargs}
        return super().pack(**self._apply_argument_scaling(kwargs))

    def pack_forget(self):
        if False:
            i = 10
            return i + 15
        ' Unmap this widget and do not use it for the packing order. '
        self._last_geometry_manager_call = None
        return super().pack_forget()

    def grid(self, **kwargs):
        if False:
            i = 10
            return i + 15
        "\n        Position a widget in the parent widget in a grid. Use as options:\n        column=number - use cell identified with given column (starting with 0)\n        columnspan=number - this widget will span several columns\n        in=master - use master to contain this widget\n        in_=master - see 'in' option description\n        ipadx=amount - add internal padding in x direction\n        ipady=amount - add internal padding in y direction\n        padx=amount - add padding in x direction\n        pady=amount - add padding in y direction\n        row=number - use cell identified with given row (starting with 0)\n        rowspan=number - this widget will span several rows\n        sticky=NSEW - if cell is larger on which sides will this widget stick to the cell boundary\n        "
        self._last_geometry_manager_call = {'function': super().grid, 'kwargs': kwargs}
        return super().grid(**self._apply_argument_scaling(kwargs))

    def grid_forget(self):
        if False:
            while True:
                i = 10
        ' Unmap this widget. '
        self._last_geometry_manager_call = None
        return super().grid_forget()