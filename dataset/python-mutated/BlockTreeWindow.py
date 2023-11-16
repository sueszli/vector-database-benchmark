"""
Copyright 2007, 2008, 2009, 2016 Free Software Foundation, Inc.
This file is part of GNU Radio

SPDX-License-Identifier: GPL-2.0-or-later

"""
from gi.repository import Gtk, Gdk, GObject
from . import Actions, Utils, Constants
(NAME_INDEX, KEY_INDEX, DOC_INDEX) = range(3)

def _format_doc(doc):
    if False:
        print('Hello World!')
    docs = []
    if doc.get(''):
        docs += doc.get('').splitlines() + ['']
    for (block_name, docstring) in doc.items():
        docs.append('--- {0} ---'.format(block_name))
        docs += docstring.splitlines()
        docs.append('')
    out = ''
    for (n, line) in enumerate(docs[:-1]):
        if n:
            out += '\n'
        out += Utils.encode(line)
        if n > 10 or len(out) > 500:
            out += '\n...'
            break
    return out or 'undocumented'

def _format_cat_tooltip(category):
    if False:
        print('Hello World!')
    tooltip = '{}: {}'.format('Category' if len(category) > 1 else 'Module', category[-1])
    if category == ('Core',):
        tooltip += '\n\nThis subtree is meant for blocks included with GNU Radio (in-tree).'
    elif category == (Constants.DEFAULT_BLOCK_MODULE_NAME,):
        tooltip += '\n\n' + Constants.DEFAULT_BLOCK_MODULE_TOOLTIP
    return tooltip

class BlockTreeWindow(Gtk.VBox):
    """The block selection panel."""
    __gsignals__ = {'create_new_block': (GObject.SignalFlags.RUN_FIRST, None, (str,))}

    def __init__(self, platform):
        if False:
            for i in range(10):
                print('nop')
        '\n        BlockTreeWindow constructor.\n        Create a tree view of the possible blocks in the platform.\n        The tree view nodes will be category names, the leaves will be block names.\n        A mouse double click or button press action will trigger the add block event.\n\n        Args:\n            platform: the particular platform will all block prototypes\n        '
        Gtk.VBox.__init__(self)
        self.platform = platform
        self.search_entry = Gtk.Entry()
        try:
            self.search_entry.set_icon_from_icon_name(Gtk.EntryIconPosition.PRIMARY, 'edit-find')
            self.search_entry.set_icon_activatable(Gtk.EntryIconPosition.PRIMARY, False)
            self.search_entry.set_icon_from_icon_name(Gtk.EntryIconPosition.SECONDARY, 'window-close')
            self.search_entry.connect('icon-release', self._handle_icon_event)
        except AttributeError:
            pass
        self.search_entry.connect('changed', self._update_search_tree)
        self.search_entry.connect('key-press-event', self._handle_search_key_press)
        self.pack_start(self.search_entry, False, False, 0)
        self.treestore = Gtk.TreeStore(GObject.TYPE_STRING, GObject.TYPE_STRING, GObject.TYPE_STRING)
        self.treestore_search = Gtk.TreeStore(GObject.TYPE_STRING, GObject.TYPE_STRING, GObject.TYPE_STRING)
        self.treeview = Gtk.TreeView(model=self.treestore)
        self.treeview.set_enable_search(False)
        self.treeview.set_search_column(-1)
        self.treeview.set_headers_visible(False)
        self.treeview.add_events(Gdk.EventMask.BUTTON_PRESS_MASK)
        self.treeview.connect('button-press-event', self._handle_mouse_button_press)
        self.treeview.connect('key-press-event', self._handle_search_key_press)
        self.treeview.get_selection().set_mode(Gtk.SelectionMode.SINGLE)
        renderer = Gtk.CellRendererText()
        column = Gtk.TreeViewColumn('Blocks', renderer, text=NAME_INDEX)
        self.treeview.append_column(column)
        self.treeview.set_tooltip_column(DOC_INDEX)
        column.set_sort_column_id(0)
        self.treestore.set_sort_column_id(0, Gtk.SortType.ASCENDING)
        self.treeview.enable_model_drag_source(Gdk.ModifierType.BUTTON1_MASK, Constants.DND_TARGETS, Gdk.DragAction.COPY)
        self.treeview.connect('drag-data-get', self._handle_drag_get_data)
        scrolled_window = Gtk.ScrolledWindow()
        scrolled_window.set_policy(Gtk.PolicyType.AUTOMATIC, Gtk.PolicyType.AUTOMATIC)
        scrolled_window.add(self.treeview)
        scrolled_window.set_size_request(Constants.DEFAULT_BLOCKS_WINDOW_WIDTH, -1)
        self.pack_start(scrolled_window, True, True, 0)
        self._categories = {tuple(): None}
        self._categories_search = {tuple(): None}
        self.platform.block_docstrings_loaded_callback = self.update_docs
        self.repopulate()

    def clear(self):
        if False:
            while True:
                i = 10
        self.treestore.clear()
        self._categories = {(): None}

    def repopulate(self):
        if False:
            print('Hello World!')
        self.clear()
        for block in self.platform.blocks.values():
            if block.category:
                self.add_block(block)
        self.expand_module_in_tree()

    def expand_module_in_tree(self, module_name='Core'):
        if False:
            while True:
                i = 10
        self.treeview.collapse_all()
        core_module_iter = self._categories.get((module_name,))
        if core_module_iter:
            self.treeview.expand_row(self.treestore.get_path(core_module_iter), False)

    def add_block(self, block, treestore=None, categories=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        Add a block with category to this selection window.\n        Add only the category when block is None.\n\n        Args:\n            block: the block object or None\n        '
        treestore = treestore or self.treestore
        categories = categories or self._categories
        category = tuple(filter(str, block.category))
        for (level, parent_cat_name) in enumerate(category, 1):
            parent_category = category[:level]
            if parent_category not in categories:
                iter_ = treestore.insert_before(categories[parent_category[:-1]], None)
                treestore.set_value(iter_, NAME_INDEX, parent_cat_name)
                treestore.set_value(iter_, KEY_INDEX, '')
                treestore.set_value(iter_, DOC_INDEX, _format_cat_tooltip(parent_category))
                categories[parent_category] = iter_
        iter_ = treestore.insert_before(categories[category], None)
        treestore.set_value(iter_, KEY_INDEX, block.key)
        treestore.set_value(iter_, NAME_INDEX, block.label)
        treestore.set_value(iter_, DOC_INDEX, _format_doc(block.documentation))

    def update_docs(self):
        if False:
            for i in range(10):
                print('nop')
        'Update the documentation column of every block'

        def update_doc(model, _, iter_):
            if False:
                i = 10
                return i + 15
            key = model.get_value(iter_, KEY_INDEX)
            if not key:
                return
            block = self.platform.blocks[key]
            model.set_value(iter_, DOC_INDEX, _format_doc(block.documentation))
        self.treestore.foreach(update_doc)
        self.treestore_search.foreach(update_doc)

    def _get_selected_block_key(self):
        if False:
            while True:
                i = 10
        '\n        Get the currently selected block key.\n\n        Returns:\n            the key of the selected block or a empty string\n        '
        selection = self.treeview.get_selection()
        (treestore, iter) = selection.get_selected()
        return iter and treestore.get_value(iter, KEY_INDEX) or ''

    def _expand_category(self):
        if False:
            for i in range(10):
                print('nop')
        (treestore, iter) = self.treeview.get_selection().get_selected()
        if iter and treestore.iter_has_child(iter):
            path = treestore.get_path(iter)
            self.treeview.expand_to_path(path)

    def _handle_icon_event(self, widget, icon, event):
        if False:
            while True:
                i = 10
        if icon == Gtk.EntryIconPosition.PRIMARY:
            pass
        elif icon == Gtk.EntryIconPosition.SECONDARY:
            widget.set_text('')
            self.search_entry.hide()

    def _update_search_tree(self, widget):
        if False:
            while True:
                i = 10
        key = widget.get_text().lower()
        if not key:
            self.treeview.set_model(self.treestore)
            self.expand_module_in_tree()
        else:
            matching_blocks = [b for b in list(self.platform.blocks.values()) if key in b.key.lower() or key in b.label.lower()]
            self.treestore_search.clear()
            self._categories_search = {tuple(): None}
            for block in matching_blocks:
                self.add_block(block, self.treestore_search, self._categories_search)
            self.treeview.set_model(self.treestore_search)
            self.treeview.expand_all()

    def _handle_search_key_press(self, widget, event):
        if False:
            i = 10
            return i + 15
        'Handle Return and Escape key events in search entry and treeview'
        if event.keyval == Gdk.KEY_Return:
            if widget == self.search_entry:
                selected = self.treestore_search.get_iter_first()
                while self.treestore_search.iter_children(selected):
                    selected = self.treestore_search.iter_children(selected)
                if selected is not None:
                    key = self.treestore_search.get_value(selected, KEY_INDEX)
                    if key:
                        self.emit('create_new_block', key)
            elif widget == self.treeview:
                key = self._get_selected_block_key()
                if key:
                    self.emit('create_new_block', key)
                else:
                    self._expand_category()
            else:
                return False
        elif event.keyval == Gdk.KEY_Escape:
            self.search_entry.set_text('')
            self.search_entry.hide()
        elif event.get_state() & Gdk.ModifierType.CONTROL_MASK and event.keyval == Gdk.KEY_f or event.keyval == Gdk.KEY_slash:
            Actions.FIND_BLOCKS.activate()
        elif event.get_state() & Gdk.ModifierType.CONTROL_MASK and event.keyval == Gdk.KEY_b:
            Actions.TOGGLE_BLOCKS_WINDOW.activate()
        else:
            return False
        return True

    def _handle_drag_get_data(self, widget, drag_context, selection_data, info, time):
        if False:
            print('Hello World!')
        '\n        Handle a drag and drop by setting the key to the selection object.\n        This will call the destination handler for drag and drop.\n        Only call set when the key is valid to ignore DND from categories.\n        '
        key = self._get_selected_block_key()
        if key:
            selection_data.set_text(key, len(key))

    def _handle_mouse_button_press(self, widget, event):
        if False:
            while True:
                i = 10
        '\n        Handle the mouse button press.\n        If a left double click is detected, call add selected block.\n        '
        if event.button == 1 and event.type == Gdk.EventType._2BUTTON_PRESS:
            key = self._get_selected_block_key()
            if key:
                self.emit('create_new_block', key)