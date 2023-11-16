from gi.repository import Gtk, GLib, Adw

@Gtk.Template(resource_path='/com/usebottles/bottles/exclusion-pattern-entry.ui')
class ExclusionPatternEntry(Adw.ActionRow):
    __gtype_name__ = 'ExclusionPatternEntry'
    btn_remove = Gtk.Template.Child()

    def __init__(self, parent, pattern, **kwargs):
        if False:
            return 10
        super().__init__(**kwargs)
        self.parent = parent
        self.manager = parent.window.manager
        self.config = parent.config
        self.pattern = pattern
        self.set_title(self.pattern)
        self.btn_remove.connect('clicked', self.__remove)

    def __remove(self, *_args):
        if False:
            i = 10
            return i + 15
        '\n        Remove the env var from the bottle configuration and\n        destroy the widget\n        '
        patterns = self.config.Versioning_Exclusion_Patterns
        if self.pattern in patterns:
            patterns.remove(self.pattern)
        self.manager.update_config(config=self.config, key='Versioning_Exclusion_Patterns', value=patterns)
        self.parent.group_patterns.remove(self)

@Gtk.Template(resource_path='/com/usebottles/bottles/dialog-exclusion-patterns.ui')
class ExclusionPatternsDialog(Adw.Window):
    __gtype_name__ = 'ExclusionPatternsDialog'
    entry_name = Gtk.Template.Child()
    group_patterns = Gtk.Template.Child()

    def __init__(self, window, config, **kwargs):
        if False:
            while True:
                i = 10
        super().__init__(**kwargs)
        self.set_transient_for(window)
        self.window = window
        self.manager = window.manager
        self.config = config
        self.__populate_patterns_list()
        self.entry_name.connect('apply', self.__save_var)

    def __save_var(self, *_args):
        if False:
            while True:
                i = 10
        '\n        This function save the new env var to the\n        bottle configuration\n        '
        pattern = self.entry_name.get_text()
        self.manager.update_config(config=self.config, key='Versioning_Exclusion_Patterns', value=self.config.Versioning_Exclusion_Patterns + [pattern])
        _entry = ExclusionPatternEntry(self, pattern)
        GLib.idle_add(self.group_patterns.add, _entry)
        self.entry_name.set_text('')

    def __populate_patterns_list(self):
        if False:
            i = 10
            return i + 15
        '\n        This function populate the list of exclusion patterns\n        with the existing ones from the bottle configuration\n        '
        patterns = self.config.Versioning_Exclusion_Patterns
        if len(patterns) == 0:
            self.group_patterns.set_description(_('No exclusion patterns defined.'))
            return
        self.group_patterns.set_description('')
        for pattern in patterns:
            _entry = ExclusionPatternEntry(self, pattern)
            GLib.idle_add(self.group_patterns.add, _entry)