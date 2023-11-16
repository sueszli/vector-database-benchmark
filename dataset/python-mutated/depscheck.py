from gi.repository import Gtk, Adw

@Gtk.Template(resource_path='/com/usebottles/bottles/dialog-deps-check.ui')
class DependenciesCheckDialog(Adw.Window):
    __gtype_name__ = 'DependenciesCheckDialog'
    btn_quit = Gtk.Template.Child()

    def __init__(self, window, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(**kwargs)
        self.set_transient_for(window)
        self.window = window
        self.btn_quit.connect('clicked', self.__quit)

    def __quit(self, *_args):
        if False:
            i = 10
            return i + 15
        self.window.proper_close()