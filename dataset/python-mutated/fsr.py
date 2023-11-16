from gi.repository import Gtk, GLib, Adw
from bottles.backend.logger import Logger
logging = Logger()

@Gtk.Template(resource_path='/com/usebottles/bottles/dialog-fsr.ui')
class FsrDialog(Adw.Window):
    __gtype_name__ = 'FsrDialog'
    btn_save = Gtk.Template.Child()
    combo_quality_mode = Gtk.Template.Child()
    str_list_quality_mode = Gtk.Template.Child()
    spin_sharpening_strength = Gtk.Template.Child()

    def __init__(self, parent_window, config, **kwargs):
        if False:
            while True:
                i = 10
        super().__init__(**kwargs)
        self.set_transient_for(parent_window)
        self.window = parent_window
        self.manager = parent_window.manager
        self.config = config
        self.quality_mode = {'none': _('None'), 'ultra': _('Ultra Quality'), 'quality': _('Quality'), 'balanced': _('Balanced'), 'performance': _('Performance')}
        self.btn_save.connect('clicked', self.__save)
        self.__update(config)

    def __update(self, config):
        if False:
            for i in range(10):
                print('nop')
        parameters = config.Parameters
        for mode in self.quality_mode.values():
            self.str_list_quality_mode.append(mode)
        if parameters.fsr_quality_mode:
            self.combo_quality_mode.set_selected(list(self.quality_mode.keys()).index(parameters.fsr_quality_mode))
        self.spin_sharpening_strength.set_value(parameters.fsr_sharpening_strength)

    def __idle_save(self, *_args):
        if False:
            return 10
        print(list(self.quality_mode.keys())[self.combo_quality_mode.get_selected()])
        settings = {'fsr_quality_mode': list(self.quality_mode.keys())[self.combo_quality_mode.get_selected()], 'fsr_sharpening_strength': int(self.spin_sharpening_strength.get_value())}
        for setting in settings.keys():
            self.manager.update_config(config=self.config, key=setting, value=settings[setting], scope='Parameters')
            self.destroy()

    def __save(self, *_args):
        if False:
            return 10
        GLib.idle_add(self.__idle_save)