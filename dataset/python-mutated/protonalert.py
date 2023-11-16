from gi.repository import Gtk, Adw

@Gtk.Template(resource_path='/com/usebottles/bottles/dialog-proton-alert.ui')
class ProtonAlertDialog(Adw.Window):
    __gtype_name__ = 'ProtonAlertDialog'
    __resources = {}
    btn_use = Gtk.Template.Child()
    btn_cancel = Gtk.Template.Child()
    check_confirm = Gtk.Template.Child()

    def __init__(self, window, callback, **kwargs):
        if False:
            print('Hello World!')
        super().__init__(**kwargs)
        self.set_transient_for(window)
        self.callback = callback
        self.btn_use.connect('clicked', self.__callback, True)
        self.btn_cancel.connect('clicked', self.__callback, False)
        self.check_confirm.connect('toggled', self.__toggle_btn_use)

    def __callback(self, _, status):
        if False:
            for i in range(10):
                print('nop')
        self.destroy()
        self.callback(status)
        self.close()

    def __toggle_btn_use(self, widget, *_args):
        if False:
            print('Hello World!')
        self.btn_use.set_sensitive(widget.get_active())