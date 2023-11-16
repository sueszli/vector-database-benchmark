from typing import Union, Tuple, List, Optional
from .core_rendering import CTkCanvas
from .theme import ThemeManager
from .core_rendering import DrawEngine
from .core_widget_classes import CTkBaseClass

class CTkFrame(CTkBaseClass):
    """
    Frame with rounded corners and border.
    Default foreground colors are set according to theme.
    To make the frame transparent set fg_color=None.
    For detailed information check out the documentation.
    """

    def __init__(self, master: any, width: int=200, height: int=200, corner_radius: Optional[Union[int, str]]=None, border_width: Optional[Union[int, str]]=None, bg_color: Union[str, Tuple[str, str]]='transparent', fg_color: Optional[Union[str, Tuple[str, str]]]=None, border_color: Optional[Union[str, Tuple[str, str]]]=None, background_corner_colors: Union[Tuple[Union[str, Tuple[str, str]]], None]=None, overwrite_preferred_drawing_method: Union[str, None]=None, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(master=master, bg_color=bg_color, width=width, height=height, **kwargs)
        self._border_color = ThemeManager.theme['CTkFrame']['border_color'] if border_color is None else self._check_color_type(border_color)
        if fg_color is None:
            if isinstance(self.master, CTkFrame):
                if self.master._fg_color == ThemeManager.theme['CTkFrame']['fg_color']:
                    self._fg_color = ThemeManager.theme['CTkFrame']['top_fg_color']
                else:
                    self._fg_color = ThemeManager.theme['CTkFrame']['fg_color']
            else:
                self._fg_color = ThemeManager.theme['CTkFrame']['fg_color']
        else:
            self._fg_color = self._check_color_type(fg_color, transparency=True)
        self._background_corner_colors = background_corner_colors
        self._corner_radius = ThemeManager.theme['CTkFrame']['corner_radius'] if corner_radius is None else corner_radius
        self._border_width = ThemeManager.theme['CTkFrame']['border_width'] if border_width is None else border_width
        self._canvas = CTkCanvas(master=self, highlightthickness=0, width=self._apply_widget_scaling(self._current_width), height=self._apply_widget_scaling(self._current_height))
        self._canvas.place(x=0, y=0, relwidth=1, relheight=1)
        self._canvas.configure(bg=self._apply_appearance_mode(self._bg_color))
        self._draw_engine = DrawEngine(self._canvas)
        self._overwrite_preferred_drawing_method = overwrite_preferred_drawing_method
        self._draw(no_color_updates=True)

    def winfo_children(self) -> List[any]:
        if False:
            while True:
                i = 10
        "\n        winfo_children of CTkFrame without self.canvas widget,\n        because it's not a child but part of the CTkFrame itself\n        "
        child_widgets = super().winfo_children()
        try:
            child_widgets.remove(self._canvas)
            return child_widgets
        except ValueError:
            return child_widgets

    def _set_scaling(self, *args, **kwargs):
        if False:
            print('Hello World!')
        super()._set_scaling(*args, **kwargs)
        self._canvas.configure(width=self._apply_widget_scaling(self._desired_width), height=self._apply_widget_scaling(self._desired_height))
        self._draw()

    def _set_dimensions(self, width=None, height=None):
        if False:
            for i in range(10):
                print('nop')
        super()._set_dimensions(width, height)
        self._canvas.configure(width=self._apply_widget_scaling(self._desired_width), height=self._apply_widget_scaling(self._desired_height))
        self._draw()

    def _draw(self, no_color_updates=False):
        if False:
            for i in range(10):
                print('nop')
        super()._draw(no_color_updates)
        if not self._canvas.winfo_exists():
            return
        if self._background_corner_colors is not None:
            self._draw_engine.draw_background_corners(self._apply_widget_scaling(self._current_width), self._apply_widget_scaling(self._current_height))
            self._canvas.itemconfig('background_corner_top_left', fill=self._apply_appearance_mode(self._background_corner_colors[0]))
            self._canvas.itemconfig('background_corner_top_right', fill=self._apply_appearance_mode(self._background_corner_colors[1]))
            self._canvas.itemconfig('background_corner_bottom_right', fill=self._apply_appearance_mode(self._background_corner_colors[2]))
            self._canvas.itemconfig('background_corner_bottom_left', fill=self._apply_appearance_mode(self._background_corner_colors[3]))
        else:
            self._canvas.delete('background_parts')
        requires_recoloring = self._draw_engine.draw_rounded_rect_with_border(self._apply_widget_scaling(self._current_width), self._apply_widget_scaling(self._current_height), self._apply_widget_scaling(self._corner_radius), self._apply_widget_scaling(self._border_width), overwrite_preferred_drawing_method=self._overwrite_preferred_drawing_method)
        if no_color_updates is False or requires_recoloring:
            if self._fg_color == 'transparent':
                self._canvas.itemconfig('inner_parts', fill=self._apply_appearance_mode(self._bg_color), outline=self._apply_appearance_mode(self._bg_color))
            else:
                self._canvas.itemconfig('inner_parts', fill=self._apply_appearance_mode(self._fg_color), outline=self._apply_appearance_mode(self._fg_color))
            self._canvas.itemconfig('border_parts', fill=self._apply_appearance_mode(self._border_color), outline=self._apply_appearance_mode(self._border_color))
            self._canvas.configure(bg=self._apply_appearance_mode(self._bg_color))

    def configure(self, require_redraw=False, **kwargs):
        if False:
            print('Hello World!')
        if 'fg_color' in kwargs:
            self._fg_color = self._check_color_type(kwargs.pop('fg_color'), transparency=True)
            require_redraw = True
            for child in self.winfo_children():
                if isinstance(child, CTkBaseClass):
                    child.configure(bg_color=self._fg_color)
        if 'bg_color' in kwargs:
            if self._fg_color == 'transparent':
                for child in self.winfo_children():
                    if isinstance(child, CTkBaseClass):
                        child.configure(bg_color=self._fg_color)
        if 'border_color' in kwargs:
            self._border_color = self._check_color_type(kwargs.pop('border_color'))
            require_redraw = True
        if 'background_corner_colors' in kwargs:
            self._background_corner_colors = kwargs.pop('background_corner_colors')
            require_redraw = True
        if 'corner_radius' in kwargs:
            self._corner_radius = kwargs.pop('corner_radius')
            require_redraw = True
        if 'border_width' in kwargs:
            self._border_width = kwargs.pop('border_width')
            require_redraw = True
        super().configure(require_redraw=require_redraw, **kwargs)

    def cget(self, attribute_name: str) -> any:
        if False:
            return 10
        if attribute_name == 'corner_radius':
            return self._corner_radius
        elif attribute_name == 'border_width':
            return self._border_width
        elif attribute_name == 'fg_color':
            return self._fg_color
        elif attribute_name == 'border_color':
            return self._border_color
        elif attribute_name == 'background_corner_colors':
            return self._background_corner_colors
        else:
            return super().cget(attribute_name)

    def bind(self, sequence=None, command=None, add=True):
        if False:
            print('Hello World!')
        ' called on the tkinter.Canvas '
        if not (add == '+' or add is True):
            raise ValueError("'add' argument can only be '+' or True to preserve internal callbacks")
        self._canvas.bind(sequence, command, add=True)

    def unbind(self, sequence=None, funcid=None):
        if False:
            while True:
                i = 10
        ' called on the tkinter.Canvas '
        if funcid is not None:
            raise ValueError("'funcid' argument can only be None, because there is a bug in" + ' tkinter and its not clear whether the internal callbacks will be unbinded or not')
        self._canvas.unbind(sequence, None)