from gi.repository import Gtk, GLib, Adw

@Gtk.Template(resource_path='/com/usebottles/bottles/dialog-gamescope.ui')
class GamescopeDialog(Adw.Window):
    __gtype_name__ = 'GamescopeDialog'
    spin_width = Gtk.Template.Child()
    spin_height = Gtk.Template.Child()
    spin_gamescope_width = Gtk.Template.Child()
    spin_gamescope_height = Gtk.Template.Child()
    spin_fps_limit = Gtk.Template.Child()
    spin_fps_limit_no_focus = Gtk.Template.Child()
    switch_scaling = Gtk.Template.Child()
    toggle_borderless = Gtk.Template.Child()
    toggle_fullscreen = Gtk.Template.Child()
    btn_save = Gtk.Template.Child()
    btn_cancel = Gtk.Template.Child()

    def __init__(self, window, config, **kwargs):
        if False:
            print('Hello World!')
        super().__init__(**kwargs)
        self.set_transient_for(window)
        self.window = window
        self.manager = window.manager
        self.config = config
        self.btn_save.connect('clicked', self.__save)
        self.toggle_borderless.connect('toggled', self.__change_wtype, 'b')
        self.toggle_fullscreen.connect('toggled', self.__change_wtype, 'f')
        self.__update(config)

    def __change_wtype(self, widget, wtype):
        if False:
            for i in range(10):
                print('nop')
        self.toggle_borderless.handler_block_by_func(self.__change_wtype)
        self.toggle_fullscreen.handler_block_by_func(self.__change_wtype)
        if wtype == 'b':
            self.toggle_fullscreen.set_active(False)
            self.toggle_borderless.set_active(True)
        elif wtype == 'f':
            self.toggle_fullscreen.set_active(True)
            self.toggle_borderless.set_active(False)
        self.toggle_borderless.handler_unblock_by_func(self.__change_wtype)
        self.toggle_fullscreen.handler_unblock_by_func(self.__change_wtype)

    def __update(self, config):
        if False:
            while True:
                i = 10
        self.toggle_borderless.handler_block_by_func(self.__change_wtype)
        self.toggle_fullscreen.handler_block_by_func(self.__change_wtype)
        parameters = config.Parameters
        self.spin_width.set_value(parameters.gamescope_game_width)
        self.spin_height.set_value(parameters.gamescope_game_height)
        self.spin_gamescope_width.set_value(parameters.gamescope_window_width)
        self.spin_gamescope_height.set_value(parameters.gamescope_window_height)
        self.spin_fps_limit.set_value(parameters.gamescope_fps)
        self.spin_fps_limit_no_focus.set_value(parameters.gamescope_fps_no_focus)
        self.switch_scaling.set_active(parameters.gamescope_scaling)
        self.toggle_borderless.set_active(parameters.gamescope_borderless)
        self.toggle_fullscreen.set_active(parameters.gamescope_fullscreen)
        self.toggle_borderless.handler_unblock_by_func(self.__change_wtype)
        self.toggle_fullscreen.handler_unblock_by_func(self.__change_wtype)

    def __idle_save(self, *_args):
        if False:
            print('Hello World!')
        settings = {'gamescope_game_width': self.spin_width.get_value(), 'gamescope_game_height': self.spin_height.get_value(), 'gamescope_window_width': self.spin_gamescope_width.get_value(), 'gamescope_window_height': self.spin_gamescope_height.get_value(), 'gamescope_fps': self.spin_fps_limit.get_value(), 'gamescope_fps_no_focus': self.spin_fps_limit_no_focus.get_value(), 'gamescope_scaling': self.switch_scaling.get_active(), 'gamescope_borderless': self.toggle_borderless.get_active(), 'gamescope_fullscreen': self.toggle_fullscreen.get_active()}
        for setting in settings.keys():
            self.manager.update_config(config=self.config, key=setting, value=settings[setting], scope='Parameters')
        self.destroy()

    def __save(self, *_args):
        if False:
            return 10
        GLib.idle_add(self.__idle_save)