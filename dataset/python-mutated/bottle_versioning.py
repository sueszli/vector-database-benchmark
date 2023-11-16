import re
from gettext import gettext as _
from gi.repository import Gtk, GLib, Adw
from bottles.backend.models.result import Result
from bottles.backend.utils.threading import RunAsync
from bottles.frontend.utils.common import open_doc_url
from bottles.frontend.utils.gtk import GtkUtils
from bottles.frontend.widgets.state import StateEntry

@Gtk.Template(resource_path='/com/usebottles/bottles/details-versioning.ui')
class VersioningView(Adw.PreferencesPage):
    __gtype_name__ = 'DetailsVersioning'
    __registry = []
    list_states = Gtk.Template.Child()
    actions = Gtk.Template.Child()
    pop_state = Gtk.Template.Child()
    btn_save = Gtk.Template.Child()
    btn_help = Gtk.Template.Child()
    entry_state_message = Gtk.Template.Child()
    status_page = Gtk.Template.Child()
    pref_page = Gtk.Template.Child()
    btn_add = Gtk.Template.Child()
    ev_controller = Gtk.EventControllerKey.new()

    def __init__(self, details, config, **kwargs):
        if False:
            i = 10
            return i + 15
        super().__init__(**kwargs)
        self.window = details.window
        self.manager = details.window.manager
        self.versioning_manager = details.window.manager.versioning_manager
        self.config = config
        self.ev_controller.connect('key-released', self.check_entry_state_message)
        self.entry_state_message.add_controller(self.ev_controller)
        self.btn_save.connect('clicked', self.add_state)
        self.btn_help.connect('clicked', open_doc_url, 'bottles/versioning')
        self.entry_state_message.connect('activate', self.add_state)

    def empty_list(self):
        if False:
            for i in range(10):
                print('nop')
        for r in self.__registry:
            if r.get_parent() is not None:
                r.get_parent().remove(r)
        self.__registry = []

    @GtkUtils.run_in_main_loop
    def update(self, widget=None, config=None, states=None, active=0):
        if False:
            return 10
        '\n        This function update the states list with the\n        ones from the bottle configuration.\n        '
        if config is None:
            config = self.config
        if states is None:
            states = self.versioning_manager.list_states(config)
            if not config.Versioning:
                active = states.data.get('state_id')
                states = states.data.get('states')
        self.config = config
        self.list_states.set_sensitive(False)
        if self.config.Versioning:
            self.btn_add.set_sensitive(False)
            self.btn_add.set_tooltip_text(_('Please migrate to the new Versioning system to create new states.'))

        def new_state(_state, active):
            if False:
                return 10
            entry = StateEntry(parent=self, config=self.config, state=_state, active=active)
            self.__registry.append(entry)
            self.list_states.append(entry)

        def callback(result, error=False):
            if False:
                while True:
                    i = 10
            self.status_page.set_visible(not result.status)
            self.pref_page.set_visible(result.status)
            self.list_states.set_visible(result.status)
            self.list_states.set_sensitive(result.status)

        def process_states():
            if False:
                while True:
                    i = 10
            GLib.idle_add(self.empty_list)
            if len(states) == 0:
                return Result(False)
            for state in states.items():
                _active = int(state[0]) == int(active)
                GLib.idle_add(new_state, state, _active)
            return Result(True)
        RunAsync(process_states, callback)

    def check_entry_state_message(self, *_args):
        if False:
            while True:
                i = 10
        '\n        This function check if the entry state message is valid,\n        looking for special characters. It also toggles the widget icon\n        and the save button sensitivity according to the result.\n        '
        regex = re.compile('[@!#$%^&*()<>?/|}{~:.;,"]')
        message = self.entry_state_message.get_text()
        check = regex.search(message) is None
        self.btn_save.set_sensitive(check)
        self.entry_state_message.set_icon_from_icon_name(1, '' if check else 'dialog-warning-symbolic"')

    def add_state(self, widget):
        if False:
            while True:
                i = 10
        '\n        This function create ask the versioning manager to\n        create a new bottle state with the given message.\n        '
        if not self.btn_save.get_sensitive():
            return

        @GtkUtils.run_in_main_loop
        def update(result, error):
            if False:
                print('Hello World!')
            self.window.show_toast(result.message)
            if result.ok:
                self.update(states=result.data.get('states'), active=result.data.get('state_id'))
        message = self.entry_state_message.get_text()
        if message != '':
            RunAsync(task_func=self.versioning_manager.create_state, callback=update, config=self.config, message=message)
            self.entry_state_message.set_text('')
            self.pop_state.popdown()