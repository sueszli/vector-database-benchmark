import subprocess
from gi.repository import Gtk, Adw, Gio
from bottles.backend.managers.manager import Manager
from bottles.backend.models.config import BottleConfig
from bottles.frontend.params import APP_ID

class BottleEntry(Adw.ActionRow):

    def __init__(self, config: BottleConfig):
        if False:
            return 10
        super().__init__()
        self.bottle = config.Path
        self.set_title(config.Name)

@Gtk.Template(resource_path='/com/usebottles/bottles/dialog-bottle-picker.ui')
class BottlePickerDialog(Adw.ApplicationWindow):
    """This class should not be called from the application GUI, only from CLI."""
    __gtype_name__ = 'BottlePickerDialog'
    settings = Gio.Settings.new(APP_ID)
    Adw.init()
    btn_cancel = Gtk.Template.Child()
    btn_select = Gtk.Template.Child()
    list_bottles = Gtk.Template.Child()
    btn_open = Gtk.Template.Child()

    def __init__(self, arg_exe, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(**kwargs)
        self.arg_exe = arg_exe
        mng = Manager(g_settings=self.settings, is_cli=True)
        mng.check_bottles()
        bottles = mng.local_bottles
        for (_, config) in bottles.items():
            self.list_bottles.append(BottleEntry(config))
        self.list_bottles.select_row(self.list_bottles.get_first_child())
        self.btn_cancel.connect('clicked', self.__close)
        self.btn_select.connect('clicked', self.__select)
        self.btn_open.connect('clicked', self.__open)

    @staticmethod
    def __close(*_args):
        if False:
            print('Hello World!')
        quit()

    def __select(self, *_args):
        if False:
            for i in range(10):
                print('nop')
        row = self.list_bottles.get_selected_row()
        if row:
            self.destroy()
            subprocess.Popen(['bottles-cli', 'run', '-b', row.bottle, '-e', self.arg_exe])

    def __open(self, *_args):
        if False:
            for i in range(10):
                print('nop')
        self.destroy()
        subprocess.Popen(['bottles'])