from gi.repository import Gtk, GLib, Adw

@Gtk.Template(resource_path='/com/usebottles/bottles/dll-override-entry.ui')
class DLLEntry(Adw.ComboRow):
    __gtype_name__ = 'DLLEntry'
    btn_remove = Gtk.Template.Child()

    def __init__(self, window, config, override, **kwargs):
        if False:
            return 10
        super().__init__(**kwargs)
        self.window = window
        self.manager = window.manager
        self.config = config
        self.override = override
        types = ('b', 'n', 'b,n', 'n,b', 'd')
        '\n        Set the DLL name as ActionRow title and set the\n        combo_type to the type of override\n        '
        self.set_title(self.override[0])
        self.set_selected(types.index(self.override[1]))
        self.btn_remove.connect('clicked', self.__remove_override)
        self.connect('notify::selected', self.__set_override_type)

    def __set_override_type(self, *_args):
        if False:
            for i in range(10):
                print('nop')
        '\n        Change the override type according to the selected\n        and update the bottle configuration\n        '
        selected = self.get_selected()
        types = ('b', 'n', 'b,n', 'n,b', 'd')
        self.manager.update_config(config=self.config, key=self.override[0], value=types[selected], scope='DLL_Overrides')

    def __remove_override(self, *_args):
        if False:
            i = 10
            return i + 15
        '\n        Remove the override from the bottle configuration and\n        destroy the widget\n        '
        self.manager.update_config(config=self.config, key=self.override[0], value=False, scope='DLL_Overrides', remove=True)
        self.get_parent().remove(self)

@Gtk.Template(resource_path='/com/usebottles/bottles/dialog-dll-overrides.ui')
class DLLOverridesDialog(Adw.PreferencesWindow):
    __gtype_name__ = 'DLLOverridesDialog'
    entry_row = Gtk.Template.Child()
    group_overrides = Gtk.Template.Child()

    def __init__(self, window, config, **kwargs):
        if False:
            return 10
        super().__init__(**kwargs)
        self.set_transient_for(window)
        self.window = window
        self.manager = window.manager
        self.config = config
        self.__populate_overrides_list()
        self.entry_row.connect('apply', self.__save_override)

    def __save_override(self, *_args):
        if False:
            print('Hello World!')
        '\n        This function check if the override name is not empty, then\n        store it in the bottle configuration and add a new entry to\n        the list. It also clears the entry field\n        '
        dll_name = self.entry_row.get_text()
        if dll_name != '':
            self.manager.update_config(config=self.config, key=dll_name, value='n,b', scope='DLL_Overrides')
            _entry = DLLEntry(window=self.window, config=self.config, override=[dll_name, 'n,b'])
            GLib.idle_add(self.group_overrides.add, _entry)
            self.group_overrides.set_description('')
            self.entry_row.set_text('')

    def __populate_overrides_list(self):
        if False:
            return 10
        '\n        This function populate the list of overrides\n        with the existing overrides from the bottle configuration\n        '
        overrides = self.config.DLL_Overrides.items()
        if len(overrides) == 0:
            self.group_overrides.set_description(_('No overrides found.'))
            return
        self.group_overrides.set_description('')
        for override in overrides:
            _entry = DLLEntry(window=self.window, config=self.config, override=override)
            GLib.idle_add(self.group_overrides.add, _entry)