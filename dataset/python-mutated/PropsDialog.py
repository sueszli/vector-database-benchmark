"""
Copyright 2007, 2008, 2009 Free Software Foundation, Inc.
This file is part of GNU Radio

SPDX-License-Identifier: GPL-2.0-or-later

"""
from gi.repository import Gtk, Gdk, GObject, Pango
from . import Actions, Utils, Constants
from .Dialogs import SimpleTextDisplay

class PropsDialog(Gtk.Dialog):
    """
    A dialog to set block parameters, view errors, and view documentation.
    """

    def __init__(self, parent, block):
        if False:
            print('Hello World!')
        '\n        Properties dialog constructor.\n\n        Args:%\n            block: a block instance\n        '
        Gtk.Dialog.__init__(self, title='Properties: ' + block.label, transient_for=parent, modal=True, destroy_with_parent=True)
        self.add_buttons(Gtk.STOCK_OK, Gtk.ResponseType.ACCEPT, Gtk.STOCK_CANCEL, Gtk.ResponseType.REJECT, Gtk.STOCK_APPLY, Gtk.ResponseType.APPLY)
        self.set_response_sensitive(Gtk.ResponseType.APPLY, False)
        self.set_size_request(*Utils.scale((Constants.MIN_DIALOG_WIDTH, Constants.MIN_DIALOG_HEIGHT)))
        self._block = block
        self._hash = 0
        self._config = parent.config
        vpaned = Gtk.VPaned()
        self.vbox.pack_start(vpaned, True, True, 0)
        notebook = self.notebook = Gtk.Notebook()
        notebook.set_show_border(False)
        notebook.set_scrollable(True)
        notebook.set_tab_pos(Gtk.PositionType.TOP)
        vpaned.pack1(notebook, True)
        self._params_boxes = []
        self._build_param_tab_boxes()
        self._docs_text_display = doc_view = SimpleTextDisplay()
        doc_view.get_buffer().create_tag('b', weight=Pango.Weight.BOLD)
        self._docs_box = Gtk.ScrolledWindow()
        self._docs_box.set_policy(Gtk.PolicyType.AUTOMATIC, Gtk.PolicyType.AUTOMATIC)
        self._docs_vbox = Gtk.VBox(homogeneous=False, spacing=0)
        self._docs_box.add(self._docs_vbox)
        self._docs_link = Gtk.Label(use_markup=True)
        self._docs_vbox.pack_start(self._docs_link, False, False, 0)
        self._docs_vbox.pack_end(self._docs_text_display, True, True, 0)
        notebook.append_page(self._docs_box, Gtk.Label(label='Documentation'))
        if Actions.TOGGLE_SHOW_CODE_PREVIEW_TAB.get_active():
            self._code_text_display = code_view = SimpleTextDisplay()
            code_view.set_wrap_mode(Gtk.WrapMode.NONE)
            code_view.get_buffer().create_tag('b', weight=Pango.Weight.BOLD)
            code_view.set_monospace(True)
            code_box = Gtk.ScrolledWindow()
            code_box.set_policy(Gtk.PolicyType.AUTOMATIC, Gtk.PolicyType.AUTOMATIC)
            code_box.add(self._code_text_display)
            notebook.append_page(code_box, Gtk.Label(label='Generated Code'))
        else:
            self._code_text_display = None
        self._error_messages_text_display = SimpleTextDisplay()
        self._error_box = Gtk.ScrolledWindow()
        self._error_box.set_policy(Gtk.PolicyType.AUTOMATIC, Gtk.PolicyType.AUTOMATIC)
        self._error_box.add(self._error_messages_text_display)
        vpaned.pack2(self._error_box)
        vpaned.set_position(int(0.65 * Constants.MIN_DIALOG_HEIGHT))
        self.connect('key-press-event', self._handle_key_press)
        self.connect('show', self.update_gui)
        self.connect('response', self._handle_response)
        self.show_all()

    def _build_param_tab_boxes(self):
        if False:
            return 10
        categories = (p.category for p in self._block.params.values())

        def unique_categories():
            if False:
                return 10
            seen = {Constants.DEFAULT_PARAM_TAB}
            yield Constants.DEFAULT_PARAM_TAB
            for cat in categories:
                if cat in seen:
                    continue
                yield cat
                seen.add(cat)
        for category in unique_categories():
            label = Gtk.Label()
            vbox = Gtk.VBox()
            scroll_box = Gtk.ScrolledWindow()
            scroll_box.set_policy(Gtk.PolicyType.AUTOMATIC, Gtk.PolicyType.AUTOMATIC)
            scroll_box.add(vbox)
            self.notebook.append_page(scroll_box, label)
            self._params_boxes.append((category, label, vbox))

    def _params_changed(self):
        if False:
            i = 10
            return i + 15
        "\n        Have the params in this dialog changed?\n        Ex: Added, removed, type change, hide change...\n        To the props dialog, the hide setting of 'none' and 'part' are identical.\n        Therefore, the props dialog only cares if the hide setting is/not 'all'.\n        Make a hash that uniquely represents the params' state.\n\n        Returns:\n            true if changed\n        "
        old_hash = self._hash
        new_hash = self._hash = hash(tuple(((hash(param), param.name, param.dtype, param.hide == 'all') for param in self._block.params.values())))
        return new_hash != old_hash

    def _handle_changed(self, *args):
        if False:
            return 10
        '\n        A change occurred within a param:\n        Rewrite/validate the block and update the gui.\n        '
        self._block.rewrite()
        self._block.validate()
        self.update_gui()

    def _activate_apply(self, *args):
        if False:
            return 10
        self.set_response_sensitive(Gtk.ResponseType.APPLY, True)

    def update_gui(self, widget=None, force=False):
        if False:
            print('Hello World!')
        '\n        Repopulate the parameters boxes (if changed).\n        Update all the input parameters.\n        Update the error messages box.\n        Hide the box if there are no errors.\n        Update the documentation block.\n        Hide the box if there are no docs.\n        '
        if force or self._params_changed():
            for (category, label, vbox) in self._params_boxes:
                vbox.hide()
                for child in vbox.get_children():
                    vbox.remove(child)
                box_all_valid = True
                force_show_id = Actions.TOGGLE_SHOW_BLOCK_IDS.get_active()
                for param in self._block.params.values():
                    if force_show_id and param.dtype == 'id':
                        param.hide = 'none'
                    if param.category != category or param.hide == 'all':
                        continue
                    box_all_valid = box_all_valid and param.is_valid()
                    input_widget = param.get_input(self._handle_changed, self._activate_apply, transient_for=self.get_transient_for())
                    input_widget.show_all()
                    vbox.pack_start(input_widget, input_widget.expand, True, 1)
                label.set_markup('<span {color}>{name}</span>'.format(color='foreground="red"' if not box_all_valid else '', name=Utils.encode(category)))
                vbox.show()
        if self._block.is_valid():
            self._error_box.hide()
        else:
            self._error_box.show()
        messages = '\n\n'.join(self._block.get_error_messages())
        self._error_messages_text_display.set_text(messages)
        self._update_docs_page()
        self._update_generated_code_page()

    def _update_docs_page(self):
        if False:
            print('Hello World!')
        'Show documentation from XML and try to display best matching docstring'
        buf = self._docs_text_display.get_buffer()
        buf.delete(buf.get_start_iter(), buf.get_end_iter())
        pos = buf.get_end_iter()
        if self._block.is_connection:
            self._docs_link.set_markup('Connection')
        elif self._block.category and self._block.category[0] == 'Core':
            note = 'Wiki Page for this Block: '
            prefix = self._config.wiki_block_docs_url_prefix
            suffix = self._block.label.replace(' ', '_')
            href = f'<a href="{prefix + suffix}">Visit Wiki Page</a>'
            self._docs_link.set_markup(href)
        else:
            self._docs_link.set_markup('Out of Tree Block')
        docstrings = self._block.documentation.copy()
        if not docstrings:
            return
        from_yaml = docstrings.pop('', '')
        for line in from_yaml.splitlines():
            if line.lstrip() == line and line.endswith(':'):
                buf.insert_with_tags_by_name(pos, line + '\n', 'b')
            else:
                buf.insert(pos, line + '\n')
        if from_yaml:
            buf.insert(pos, '\n')
        block_templates = getattr(self._block, 'templates', None)
        if block_templates:
            block_constructor = block_templates.render('make').rsplit('.', 2)[-1]
            block_class = block_constructor.partition('(')[0].strip()
            if block_class in docstrings:
                docstrings = {block_class: docstrings[block_class]}
        for (cls_name, docstring) in docstrings.items():
            buf.insert_with_tags_by_name(pos, cls_name + '\n', 'b')
            buf.insert(pos, docstring + '\n\n')
        pos.backward_chars(2)
        buf.delete(pos, buf.get_end_iter())

    def _update_generated_code_page(self):
        if False:
            while True:
                i = 10
        if not self._code_text_display:
            return
        buf = self._code_text_display.get_buffer()
        block = self._block
        key = block.key
        if key == 'epy_block':
            src = block.params['_source_code'].get_value()
        elif key == 'epy_module':
            src = block.params['source_code'].get_value()
        else:
            src = ''

        def insert(header, text):
            if False:
                while True:
                    i = 10
            if not text:
                return
            buf.insert_with_tags_by_name(buf.get_end_iter(), header, 'b')
            buf.insert(buf.get_end_iter(), text)
        buf.delete(buf.get_start_iter(), buf.get_end_iter())
        insert('# Imports\n', block.templates.render('imports').strip('\n'))
        if block.is_variable:
            insert('\n\n# Variables\n', block.templates.render('var_make'))
        insert('\n\n# Blocks\n', block.templates.render('make'))
        if src:
            insert('\n\n# External Code ({}.py)\n'.format(block.name), src)

    def _handle_key_press(self, widget, event):
        if False:
            print('Hello World!')
        close_dialog = event.keyval == Gdk.KEY_Return and event.get_state() & Gdk.ModifierType.CONTROL_MASK == 0 and (not isinstance(widget.get_focus(), Gtk.TextView))
        if close_dialog:
            self.response(Gtk.ResponseType.ACCEPT)
            return True
        return False

    def _handle_response(self, widget, response):
        if False:
            while True:
                i = 10
        if response in (Gtk.ResponseType.APPLY, Gtk.ResponseType.ACCEPT):
            for (tab, label, vbox) in self._params_boxes:
                for child in vbox.get_children():
                    child.apply_pending_changes()
            self.set_response_sensitive(Gtk.ResponseType.APPLY, False)
            return True
        return False