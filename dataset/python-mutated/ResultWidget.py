from __future__ import annotations
import logging
from html import unescape
from types import SimpleNamespace
from typing import Any
from gi.repository import Gtk, Pango
from ulauncher.api.shared.query import Query
from ulauncher.utils.icon import load_icon_surface
from ulauncher.utils.Settings import Settings
from ulauncher.utils.text_highlighter import highlight_text
from ulauncher.utils.wm import get_text_scaling_factor
ELLIPSIZE_MIN_LENGTH = 6
ELLIPSIZE_FORCE_AT_LENGTH = 20
logger = logging.getLogger()

class ResultWidget(Gtk.EventBox):
    __gtype_name__ = 'ResultWidget'
    index: int = 0
    builder: Any
    name: str
    query: Query
    result: Any
    item_box: Any
    compact = False

    def initialize(self, builder: Any, result: Any, index: int, query: Query) -> None:
        if False:
            while True:
                i = 10
        self.builder = builder
        item_frame = self.builder.get_object('item-frame')
        item_frame.connect('button-release-event', self.on_click)
        item_frame.connect('enter_notify_event', self.on_mouse_hover)
        self.item_box = builder.get_object('item-box')
        self.result = result
        self.compact = result.compact
        self.query = query
        self.set_index(index)
        text_scaling_factor = get_text_scaling_factor()
        item_container = builder.get_object('item-container')
        item_container.get_style_context().add_class('small-result-item')
        item_name = builder.get_object('name_wrapper')
        sizes = SimpleNamespace(icon=40, inner_margin_x=12 * text_scaling_factor, outer_margin_x=18 * text_scaling_factor, margin_y=5 * text_scaling_factor)
        if self.compact:
            sizes.icon = 25
            sizes.margin_y = 3 * text_scaling_factor
        item_container.set_property('margin-start', sizes.outer_margin_x)
        item_container.set_property('margin-end', sizes.outer_margin_x)
        item_container.set_property('margin-top', sizes.margin_y)
        item_container.set_property('margin-bottom', sizes.margin_y)
        item_name.set_property('margin-start', sizes.inner_margin_x)
        item_name.set_property('margin-end', sizes.inner_margin_x)
        item_name.set_property('width-request', 350 * text_scaling_factor)
        self.set_icon(load_icon_surface(result.icon, sizes.icon, self.get_scale_factor()))
        self.set_description(result.get_description(query))
        self.highlight_name()

    def set_index(self, index: int):
        if False:
            for i in range(10):
                print('nop')
        '\n        Set index for the item and assign shortcut\n        '
        jump_keys = Settings.load().get_jump_keys()
        if index < len(jump_keys):
            self.index = index
            self.set_shortcut(f'Alt+{jump_keys[index]}')

    def select(self):
        if False:
            while True:
                i = 10
        self.item_box.get_style_context().add_class('selected')
        self.scroll_to_focus()

    def deselect(self):
        if False:
            while True:
                i = 10
        self.item_box.get_style_context().remove_class('selected')

    def scroll_to_focus(self):
        if False:
            i = 10
            return i + 15
        viewport = self.item_box.get_ancestor(Gtk.Viewport)
        viewport_height = viewport.get_allocation().height
        scroll_y = viewport.get_vadjustment().get_value()
        allocation = self.get_allocation()
        bottom = allocation.y + allocation.height
        if scroll_y > allocation.y:
            viewport.set_vadjustment(Gtk.Adjustment(allocation.y, 0, 2 ** 32, 1, 10, 0))
        elif viewport_height + scroll_y < bottom:
            viewport.set_vadjustment(Gtk.Adjustment(bottom - viewport_height, 0, 2 ** 32, 1, 10, 0))

    def set_icon(self, icon):
        if False:
            i = 10
            return i + 15
        '\n        :param PixBuf icon:\n        '
        if icon:
            self.builder.get_object('item-icon').set_from_surface(icon)

    def highlight_name(self) -> None:
        if False:
            print('Hello World!')
        item = self.builder.get_object('item-name')
        highlightable_input = self.result.get_highlightable_input(self.query)
        if highlightable_input and (self.result.searchable or self.result.highlightable):
            labels = []
            for (label_text, is_highlight) in highlight_text(highlightable_input, self.result.name):
                ellipsize_min = not is_highlight and ELLIPSIZE_MIN_LENGTH or ELLIPSIZE_FORCE_AT_LENGTH
                ellipsize = Pango.EllipsizeMode.MIDDLE if len(label_text) > ellipsize_min else Pango.EllipsizeMode.NONE
                label = Gtk.Label(label=unescape(label_text), ellipsize=ellipsize)
                if is_highlight:
                    label.get_style_context().add_class('item-highlight')
                labels.append(label)
        else:
            labels = [Gtk.Label(label=self.result.name, ellipsize=Pango.EllipsizeMode.MIDDLE)]
        for label in labels:
            item.pack_start(label, False, False, 0)

    def on_click(self, _widget, event=None):
        if False:
            return 10
        window = self.get_toplevel()
        window.select_result(self.index)
        alt = bool(event and event.button != 1)
        window.handle_event(window.results_nav.activate(self.query, alt=alt))

    def on_mouse_hover(self, _widget, event):
        if False:
            return 10
        if event.time:
            self.get_toplevel().select_result(self.index)

    def set_description(self, description):
        if False:
            i = 10
            return i + 15
        description_obj = self.builder.get_object('item-descr')
        if description and (not self.compact):
            description_obj.set_text(unescape(description))
        else:
            description_obj.destroy()

    def set_shortcut(self, text):
        if False:
            print('Hello World!')
        self.builder.get_object('item-shortcut').set_text(text)

    def get_keyword(self):
        if False:
            for i in range(10):
                print('nop')
        return self.result.keyword