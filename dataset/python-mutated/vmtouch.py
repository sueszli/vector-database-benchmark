from gi.repository import Gtk, GLib, Adw

@Gtk.Template(resource_path='/com/usebottles/bottles/dialog-vmtouch.ui')
class VmtouchDialog(Adw.Window):
    __gtype_name__ = 'VmtouchDialog'
    switch_cache_cwd = Gtk.Template.Child()
    btn_save = Gtk.Template.Child()
    btn_cancel = Gtk.Template.Child()

    def __init__(self, window, config, **kwargs):
        if False:
            return 10
        super().__init__(**kwargs)
        self.set_transient_for(window)
        self.window = window
        self.manager = window.manager
        self.config = config
        self.btn_save.connect('clicked', self.__save)
        self.__update(config)

    def __update(self, config):
        if False:
            while True:
                i = 10
        self.switch_cache_cwd.set_active(config.Parameters.vmtouch_cache_cwd)

    def __idle_save(self, *_args):
        if False:
            print('Hello World!')
        settings = {'vmtouch_cache_cwd': self.switch_cache_cwd.get_active()}
        for setting in settings.keys():
            self.manager.update_config(config=self.config, key=setting, value=settings[setting], scope='Parameters')
        self.destroy()

    def __save(self, *_args):
        if False:
            for i in range(10):
                print('nop')
        GLib.idle_add(self.__idle_save)