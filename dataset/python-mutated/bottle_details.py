import os
import uuid
from datetime import datetime
from gettext import gettext as _
from typing import List, Optional
from gi.repository import Gtk, Gio, Adw, Gdk, GLib
from bottles.backend.managers.backup import BackupManager
from bottles.backend.models.config import BottleConfig
from bottles.backend.utils.manager import ManagerUtils
from bottles.backend.utils.terminal import TerminalUtils
from bottles.backend.utils.threading import RunAsync
from bottles.backend.wine.cmd import CMD
from bottles.backend.wine.control import Control
from bottles.backend.wine.executor import WineExecutor
from bottles.backend.wine.explorer import Explorer
from bottles.backend.wine.regedit import Regedit
from bottles.backend.wine.taskmgr import Taskmgr
from bottles.backend.wine.uninstaller import Uninstaller
from bottles.backend.wine.wineboot import WineBoot
from bottles.backend.wine.winecfg import WineCfg
from bottles.backend.wine.winedbg import WineDbg
from bottles.backend.wine.wineserver import WineServer
from bottles.frontend.utils.common import open_doc_url
from bottles.frontend.utils.filters import add_executable_filters, add_all_filters
from bottles.frontend.utils.gtk import GtkUtils
from bottles.frontend.widgets.program import ProgramEntry
from bottles.frontend.windows.duplicate import DuplicateDialog
from bottles.frontend.windows.upgradeversioning import UpgradeVersioningDialog

@Gtk.Template(resource_path='/com/usebottles/bottles/details-bottle.ui')
class BottleView(Adw.PreferencesPage):
    __gtype_name__ = 'DetailsBottle'
    __registry = []
    label_runner = Gtk.Template.Child()
    label_state = Gtk.Template.Child()
    label_environment = Gtk.Template.Child()
    label_arch = Gtk.Template.Child()
    install_programs = Gtk.Template.Child()
    add_shortcuts = Gtk.Template.Child()
    btn_execute = Gtk.Template.Child()
    popover_exec_settings = Gtk.Template.Child()
    exec_arguments = Gtk.Template.Child()
    exec_terminal = Gtk.Template.Child()
    row_winecfg = Gtk.Template.Child()
    row_preferences = Gtk.Template.Child()
    row_dependencies = Gtk.Template.Child()
    row_snapshots = Gtk.Template.Child()
    row_taskmanager = Gtk.Template.Child()
    row_debug = Gtk.Template.Child()
    row_explorer = Gtk.Template.Child()
    row_cmd = Gtk.Template.Child()
    row_taskmanager_legacy = Gtk.Template.Child()
    row_controlpanel = Gtk.Template.Child()
    row_uninstaller = Gtk.Template.Child()
    row_regedit = Gtk.Template.Child()
    btn_shutdown = Gtk.Template.Child()
    btn_reboot = Gtk.Template.Child()
    btn_browse = Gtk.Template.Child()
    btn_forcestop = Gtk.Template.Child()
    btn_update = Gtk.Template.Child()
    btn_toggle_removed = Gtk.Template.Child()
    btn_backup_config = Gtk.Template.Child()
    btn_backup_full = Gtk.Template.Child()
    btn_duplicate = Gtk.Template.Child()
    btn_delete = Gtk.Template.Child()
    btn_flatpak_doc = Gtk.Template.Child()
    label_name = Gtk.Template.Child()
    dot_versioning = Gtk.Template.Child()
    grid_versioning = Gtk.Template.Child()
    group_programs = Gtk.Template.Child()
    actions = Gtk.Template.Child()
    row_no_programs = Gtk.Template.Child()
    bottom_bar = Gtk.Template.Child()
    drop_overlay = Gtk.Template.Child()
    content = Gdk.ContentFormats.new_for_gtype(Gdk.FileList)
    target = Gtk.DropTarget(formats=content, actions=Gdk.DragAction.COPY)
    style_provider = Gtk.CssProvider()

    def __init__(self, details, config, **kwargs):
        if False:
            print('Hello World!')
        super().__init__(**kwargs)
        self.window = details.window
        self.manager = details.window.manager
        self.stack_bottle = details.stack_bottle
        self.leaflet = details.leaflet
        self.details = details
        self.config = config
        self.show_hidden = False
        self.target.connect('drop', self.on_drop)
        self.add_controller(self.target)
        self.target.connect('enter', self.on_enter)
        self.target.connect('leave', self.on_leave)
        self.add_shortcuts.connect('clicked', self.add)
        self.install_programs.connect('clicked', self.__change_page, 'installers')
        self.btn_execute.connect('clicked', self.run_executable)
        self.popover_exec_settings.connect('closed', self.__run_executable_with_args)
        self.row_preferences.connect('activated', self.__change_page, 'preferences')
        self.row_dependencies.connect('activated', self.__change_page, 'dependencies')
        self.row_snapshots.connect('activated', self.__change_page, 'versioning')
        self.row_taskmanager.connect('activated', self.__change_page, 'taskmanager')
        self.row_winecfg.connect('activated', self.run_winecfg)
        self.row_debug.connect('activated', self.run_debug)
        self.row_explorer.connect('activated', self.run_explorer)
        self.row_cmd.connect('activated', self.run_cmd)
        self.row_taskmanager_legacy.connect('activated', self.run_taskmanager)
        self.row_controlpanel.connect('activated', self.run_controlpanel)
        self.row_uninstaller.connect('activated', self.run_uninstaller)
        self.row_regedit.connect('activated', self.run_regedit)
        self.btn_browse.connect('clicked', self.run_browse)
        self.btn_delete.connect('clicked', self.__confirm_delete)
        self.btn_shutdown.connect('clicked', self.wineboot, 2)
        self.btn_reboot.connect('clicked', self.wineboot, 1)
        self.btn_forcestop.connect('clicked', self.wineboot, 0)
        self.btn_update.connect('clicked', self.__scan_programs)
        self.btn_toggle_removed.connect('clicked', self.__toggle_removed)
        self.btn_backup_config.connect('clicked', self.__backup, 'config')
        self.btn_backup_full.connect('clicked', self.__backup, 'full')
        self.btn_duplicate.connect('clicked', self.__duplicate)
        self.btn_flatpak_doc.connect('clicked', open_doc_url, 'flatpak/black-screen-or-silent-crash')
        if 'FLATPAK_ID' in os.environ:
            '\n            If Flatpak, show the btn_flatpak_doc widget to reach\n            the documentation on how to expose directories\n            '
            self.btn_flatpak_doc.set_visible(True)

    def __change_page(self, _widget, page_name):
        if False:
            for i in range(10):
                print('nop')
        '\n        This function try to change the page based on user choice, if\n        the page is not available, it will show the "bottle" page.\n        '
        if page_name == 'taskmanager':
            self.details.view_taskmanager.update(config=self.config)
        try:
            self.stack_bottle.set_visible_child_name(page_name)
            self.leaflet.navigate(Adw.NavigationDirection.FORWARD)
        except:
            pass

    def on_drop(self, drop_target, value: Gdk.FileList, x, y, user_data=None):
        if False:
            i = 10
            return i + 15
        self.drop_overlay.set_visible(False)
        files: List[Gio.File] = value.get_files()
        args = ''
        file = files[0]
        if '.exe' in file.get_basename().split('/')[-1] or '.msi' in file.get_basename().split('/')[-1]:
            executor = WineExecutor(self.config, exec_path=file.get_path(), args=args, terminal=self.config.run_in_terminal)

            def callback(a, b):
                if False:
                    while True:
                        i = 10
                self.update_programs()
            RunAsync(executor.run, callback)
        else:
            self.window.show_toast(_('File "{0}" is not a .exe or .msi file').format(file.get_basename().split('/')[-1]))

    def on_enter(self, drop_target, x, y):
        if False:
            for i in range(10):
                print('nop')
        self.drop_overlay.set_visible(True)
        return Gdk.DragAction.COPY

    def on_leave(self, drop_target):
        if False:
            return 10
        self.drop_overlay.set_visible(False)

    def set_config(self, config: BottleConfig):
        if False:
            print('Hello World!')
        self.config = config
        self.__update_by_env()
        update_date = datetime.strptime(self.config.Update_Date, '%Y-%m-%d %H:%M:%S.%f')
        update_date = update_date.strftime('%b %d %Y %H:%M:%S')
        self.label_name.set_tooltip_text(_('Updated: %s' % update_date))
        self.label_arch.set_text((self.config.Arch or 'n/a').capitalize())
        self.label_name.set_text(self.config.Name)
        self.label_runner.set_text(self.config.Runner)
        self.label_environment.set_text(_(self.config.Environment))
        self.dot_versioning.set_visible(self.config.Versioning)
        self.grid_versioning.set_visible(self.config.Versioning)
        self.label_state.set_text(str(self.config.State))
        self.__set_steam_rules()
        if config.Versioning:
            self.__upgrade_versioning()
        if config.Runner not in self.manager.runners_available and (not self.config.Environment == 'Steam'):
            self.__alert_missing_runner()
        self.update_programs()

    def add(self, widget=False):
        if False:
            i = 10
            return i + 15
        '\n        This function popup the add program dialog to the user. It\n        will also update the bottle configuration, appending the\n        path to the program picked by the user.\n        The file chooser path is set to the bottle path by default.\n        '

        def set_path(_dialog, response):
            if False:
                print('Hello World!')
            if response != Gtk.ResponseType.ACCEPT:
                return
            path = dialog.get_file().get_path()
            basename = dialog.get_file().get_basename()
            _uuid = str(uuid.uuid4())
            _program = {'executable': basename, 'name': basename[:-4], 'path': path, 'id': _uuid, 'folder': ManagerUtils.get_exe_parent_dir(self.config, path)}
            self.config = self.manager.update_config(config=self.config, key=_uuid, value=_program, scope='External_Programs', fallback=True).data['config']
            self.update_programs(config=self.config, force_add=_program)
            self.window.show_toast(_('"{0}" added').format(basename[:-4]))
        dialog = Gtk.FileChooserNative.new(title=_('Select Executable'), action=Gtk.FileChooserAction.OPEN, parent=self.window, accept_label=_('Add'))
        add_executable_filters(dialog)
        dialog.set_modal(True)
        dialog.connect('response', set_path)
        dialog.show()

    def update_programs(self, config: Optional[BottleConfig]=None, force_add: dict=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        This function update the programs lists.\n        '
        if config:
            if not isinstance(config, BottleConfig):
                raise TypeError('config param need BottleConfig type, but it was %s' % type(config))
            self.config = config
        if not force_add:
            GLib.idle_add(self.empty_list)

        def new_program(_program, check_boot=None, is_steam=False, wineserver_status=False):
            if False:
                while True:
                    i = 10
            if check_boot is None:
                check_boot = wineserver_status
            self.add_program(ProgramEntry(self.window, self.config, _program, is_steam=is_steam, check_boot=check_boot))
        if force_add:
            wineserver_status = WineServer(self.config).is_alive()
            new_program(force_add, None, False, wineserver_status)
            return

        def process_programs():
            if False:
                i = 10
                return i + 15
            wineserver_status = WineServer(self.config).is_alive()
            programs = self.manager.get_programs(self.config)
            handled = 0
            if self.config.Environment == 'Steam':
                GLib.idle_add(new_program, {'name': self.config.Name}, None, True)
                handled += 1
            for program in programs:
                if program.get('removed'):
                    if self.show_hidden:
                        GLib.idle_add(new_program, program, None, False, wineserver_status)
                        handled += 1
                    continue
                GLib.idle_add(new_program, program, None, False, wineserver_status)
                handled += 1
            self.row_no_programs.set_visible(handled == 0)
        process_programs()

    def add_program(self, widget):
        if False:
            print('Hello World!')
        self.__registry.append(widget)
        self.group_programs.remove(self.bottom_bar)
        self.group_programs.add(widget)
        self.group_programs.add(self.bottom_bar)

    def __toggle_removed(self, widget=False):
        if False:
            i = 10
            return i + 15
        '\n        This function toggle the show_hidden variable.\n        '
        if self.show_hidden:
            self.btn_toggle_removed.set_property('text', _('Show Hidden Programs'))
        else:
            self.btn_toggle_removed.set_property('text', _('Hide Hidden Programs'))
        self.show_hidden = not self.show_hidden
        self.update_programs(config=self.config)

    def __scan_programs(self, widget=False):
        if False:
            return 10
        self.update_programs(config=self.config)

    def empty_list(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        This function empty the programs list.\n        '
        for r in self.__registry:
            self.group_programs.remove(r)
        self.__registry = []

    def __run_executable_with_args(self, widget):
        if False:
            i = 10
            return i + 15
        '\n        This function saves updates the run arguments for the current session.\n        '
        args = self.exec_arguments.get_text()
        self.config.session_arguments = args
        self.config.run_in_terminal = self.exec_terminal.get_active()

    def run_executable(self, widget, args=False):
        if False:
            i = 10
            return i + 15
        '\n        This function pop up the dialog to run an executable.\n        The file will be executed by the runner after the\n        user confirmation.\n        '

        def show_chooser(*_args):
            if False:
                while True:
                    i = 10
            self.window.settings.set_boolean('show-sandbox-warning', False)

            def execute(_dialog, response):
                if False:
                    while True:
                        i = 10
                if response != Gtk.ResponseType.ACCEPT:
                    return
                self.window.show_toast(_('Launching "{0}"…').format(dialog.get_file().get_basename()))
                executor = WineExecutor(self.config, exec_path=dialog.get_file().get_path(), args=self.config.get('session_arguments'), terminal=self.config.get('run_in_terminal'))

                def callback(a, b):
                    if False:
                        for i in range(10):
                            print('nop')
                    self.update_programs()
                RunAsync(executor.run, callback)
            dialog = Gtk.FileChooserNative.new(title=_('Select Executable'), action=Gtk.FileChooserAction.OPEN, parent=self.window, accept_label=_('Run'))
            add_executable_filters(dialog)
            add_all_filters(dialog)
            dialog.set_modal(True)
            dialog.connect('response', execute)
            dialog.show()
        if 'FLATPAK_ID' in os.environ and self.window.settings.get_boolean('show-sandbox-warning'):
            dialog = Adw.MessageDialog.new(self.window, _('Be Aware of Sandbox'), _("Bottles is running in a sandbox, a restricted permission environment needed to keep you safe. If the program won't run, consider moving inside the bottle (3 dots icon on the top), then launch from there."))
            dialog.add_response('ok', _('_Dismiss'))
            dialog.connect('response', show_chooser)
            dialog.present()
        else:
            show_chooser()

    def __backup(self, widget, backup_type):
        if False:
            i = 10
            return i + 15
        '\n        This function pop up the file chooser where the user\n        can select the path where to export the bottle backup.\n        Use the backup_type param to export config or full.\n        '
        if backup_type == 'config':
            title = _('Select the location where to save the backup config')
            hint = f'backup_{self.config.Path}.yml'
            accept_label = _('Export')
        else:
            title = _('Select the location where to save the backup archive')
            hint = f'backup_{self.config.Path}.tar.gz'
            accept_label = _('Backup')

        @GtkUtils.run_in_main_loop
        def finish(result, error=False):
            if False:
                return 10
            if result.ok:
                self.window.show_toast(_('Backup created for "{0}"').format(self.config.Name))
            else:
                self.window.show_toast(_('Backup failed for "{0}"').format(self.config.Name))

        def set_path(_dialog, response):
            if False:
                print('Hello World!')
            if response != Gtk.ResponseType.ACCEPT:
                return
            path = dialog.get_file().get_path()
            RunAsync(task_func=BackupManager.export_backup, callback=finish, config=self.config, scope=backup_type, path=path)
        dialog = Gtk.FileChooserNative.new(title=title, action=Gtk.FileChooserAction.SAVE, parent=self.window, accept_label=accept_label)
        dialog.set_modal(True)
        dialog.connect('response', set_path)
        dialog.set_current_name(hint)
        dialog.show()

    def __duplicate(self, widget):
        if False:
            while True:
                i = 10
        '\n        This function pop up the duplicate dialog, so the user can\n        choose the new bottle name and perform duplication.\n        '
        new_window = DuplicateDialog(self)
        new_window.present()

    def __upgrade_versioning(self):
        if False:
            while True:
                i = 10
        '\n        This function pop up the upgrade versioning dialog, so the user can\n        upgrade the versioning system from old Bottles built-in to FVS.\n        '
        new_window = UpgradeVersioningDialog(self)
        new_window.present()

    def __confirm_delete(self, widget):
        if False:
            i = 10
            return i + 15
        '\n        This function pop up to delete confirm dialog. If user confirm\n        it will ask the manager to delete the bottle and will return\n        to the bottles list.\n        '

        def handle_response(_widget, response_id):
            if False:
                i = 10
                return i + 15
            if response_id == 'ok':
                RunAsync(self.manager.delete_bottle, config=self.config)
                self.window.page_list.disable_bottle(self.config)
            _widget.destroy()
        dialog = Adw.MessageDialog.new(self.window, _('Are you sure you want to permanently delete "{}"?'.format(self.config['Name'])), _('This will permanently delete all programs and settings associated with it.'))
        dialog.add_response('cancel', _('_Cancel'))
        dialog.add_response('ok', _('_Delete'))
        dialog.set_response_appearance('ok', Adw.ResponseAppearance.DESTRUCTIVE)
        dialog.connect('response', handle_response)
        dialog.present()

    def __alert_missing_runner(self):
        if False:
            print('Hello World!')
        '\n        This function pop up a dialog which alert the user that the runner\n        specified in the bottle configuration is missing.\n        '

        def handle_response(_widget, response_id):
            if False:
                print('Hello World!')
            _widget.destroy()
        dialog = Adw.MessageDialog.new(self.window, _('Missing Runner'), _('The runner requested by this bottle is missing. Install it through the Bottles preferences or choose a new one to run applications.'))
        dialog.add_response('ok', _('_Dismiss'))
        dialog.connect('response', handle_response)
        dialog.present()

    def __update_by_env(self):
        if False:
            i = 10
            return i + 15
        widgets = [self.row_uninstaller, self.row_regedit]
        for widget in widgets:
            widget.set_visible(True)
    '\n    The following functions are used like wrappers for the\n    runner utilities.\n    '

    def run_winecfg(self, widget):
        if False:
            i = 10
            return i + 15
        program = WineCfg(self.config)
        RunAsync(program.launch)

    def run_debug(self, widget):
        if False:
            i = 10
            return i + 15
        program = WineDbg(self.config)
        RunAsync(program.launch_terminal)

    def run_browse(self, widget):
        if False:
            print('Hello World!')
        ManagerUtils.open_filemanager(self.config)

    def run_explorer(self, widget):
        if False:
            for i in range(10):
                print('nop')
        program = Explorer(self.config)
        RunAsync(program.launch)

    def run_cmd(self, widget):
        if False:
            while True:
                i = 10
        program = CMD(self.config)
        RunAsync(program.launch_terminal)

    @staticmethod
    def run_snake(widget, event):
        if False:
            return 10
        if event.button == 2:
            RunAsync(TerminalUtils().launch_snake)

    def run_taskmanager(self, widget):
        if False:
            while True:
                i = 10
        program = Taskmgr(self.config)
        RunAsync(program.launch)

    def run_controlpanel(self, widget):
        if False:
            while True:
                i = 10
        program = Control(self.config)
        RunAsync(program.launch)

    def run_uninstaller(self, widget):
        if False:
            while True:
                i = 10
        program = Uninstaller(self.config)
        RunAsync(program.launch)

    def run_regedit(self, widget):
        if False:
            i = 10
            return i + 15
        program = Regedit(self.config)
        RunAsync(program.launch)

    def wineboot(self, widget, status):
        if False:
            print('Hello World!')

        @GtkUtils.run_in_main_loop
        def reset(result=None, error=False):
            if False:
                for i in range(10):
                    print('nop')
            widget.set_sensitive(True)

        def handle_response(_widget, response_id):
            if False:
                for i in range(10):
                    print('nop')
            if response_id == 'ok':
                RunAsync(wineboot.send_status, callback=reset, status=status)
            else:
                reset()
            _widget.destroy()
        wineboot = WineBoot(self.config)
        widget.set_sensitive(False)
        if status == 0:
            dialog = Adw.MessageDialog.new(self.window, _('Are you sure you want to force stop all processes?'), _('This can cause data loss, corruption, and programs to malfunction.'))
            dialog.add_response('cancel', _('_Cancel'))
            dialog.add_response('ok', _('Force _Stop'))
            dialog.set_response_appearance('ok', Adw.ResponseAppearance.DESTRUCTIVE)
            dialog.connect('response', handle_response)
            dialog.present()

    def __set_steam_rules(self):
        if False:
            i = 10
            return i + 15
        status = False if self.config.Environment == 'Steam' else True
        for w in [self.btn_delete, self.btn_backup_full, self.btn_duplicate]:
            w.set_visible(status)
            w.set_sensitive(status)