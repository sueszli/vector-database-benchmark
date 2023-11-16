from typing import Optional
from gi.repository import Gtk
from bottles.backend.models.config import BottleConfig
from bottles.backend.utils.threading import RunAsync
from bottles.backend.wine.winebridge import WineBridge
from bottles.backend.wine.winedbg import WineDbg
from bottles.frontend.utils.gtk import GtkUtils

@Gtk.Template(resource_path='/com/usebottles/bottles/details-taskmanager.ui')
class TaskManagerView(Gtk.ScrolledWindow):
    __gtype_name__ = 'TaskManagerView'
    treeview_processes = Gtk.Template.Child()
    actions = Gtk.Template.Child()
    btn_update = Gtk.Template.Child()
    btn_kill = Gtk.Template.Child()

    def __init__(self, details, config, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(**kwargs)
        self.window = details.window
        self.manager = details.window.manager
        self.config = config
        self.btn_update.connect('clicked', self.sensitive_update)
        self.btn_kill.connect('clicked', self.kill_process)
        self.treeview_processes.connect('cursor-changed', self.show_kill_btn)
        self.liststore_processes = Gtk.ListStore(str, str, str)
        self.treeview_processes.set_model(self.liststore_processes)
        cell_renderer = Gtk.CellRendererText()
        i = 0
        for column in ['PID', 'Name', 'Threads']:
            '\n            For each column, add it to the treeview_processes\n            '
            column = Gtk.TreeViewColumn(column, cell_renderer, text=i)
            self.treeview_processes.append_column(column)
            i += 1
        self.update()

    def set_config(self, config):
        if False:
            for i in range(10):
                print('nop')
        self.config = config

    def show_kill_btn(self, widget):
        if False:
            return 10
        selected = self.treeview_processes.get_selection()
        (model, treeiter) = selected.get_selected()
        if model is None or len(model) == 0:
            self.btn_kill.set_sensitive(False)
            return
        self.btn_kill.set_sensitive(True)

    def update(self, widget=False, config: Optional[BottleConfig]=None):
        if False:
            print('Hello World!')
        '\n        This function scan for new processed and update the\n        liststore_processes with the new data\n        '
        self.liststore_processes.clear()

        def fetch_processes(config: Optional[BottleConfig]=None):
            if False:
                for i in range(10):
                    print('nop')
            if config is None:
                config = BottleConfig()
            self.config = config
            if not config.Runner:
                return []
            winebridge = WineBridge(config)
            if winebridge.is_available():
                processes = winebridge.get_procs()
            else:
                winedbg = WineDbg(config)
                processes = winedbg.get_processes()
            return processes

        def update_processes(processes: list, *_args):
            if False:
                return 10
            if len(processes) > 0:
                for process in processes:
                    self.liststore_processes.append([process.get('pid'), process.get('name', 'n/a'), process.get('threads', '0')])
        RunAsync(task_func=fetch_processes, callback=update_processes, config=config)

    def sensitive_update(self, widget):
        if False:
            while True:
                i = 10

        @GtkUtils.run_in_main_loop
        def reset(result, error):
            if False:
                return 10
            self.btn_update.set_sensitive(True)
        self.btn_update.set_sensitive(False)
        RunAsync(task_func=self.update, callback=reset, widget=False, config=self.config)

    def kill_process(self, widget):
        if False:
            return 10
        winebridge = WineBridge(self.config)
        selected = self.treeview_processes.get_selection()
        (model, treeiter) = selected.get_selected()
        if model is None:
            self.btn_kill.set_sensitive(False)
            return
        pid = model[treeiter][0]
        self.btn_kill.set_sensitive(False)

        @GtkUtils.run_in_main_loop
        def reset(result, error):
            if False:
                while True:
                    i = 10
            self.liststore_processes.remove(treeiter)
        if winebridge.is_available():
            RunAsync(task_func=winebridge.kill_proc, callback=reset, pid=pid)
        else:
            winedbg = WineDbg(self.config)
            RunAsync(task_func=winedbg.kill_process, callback=reset, pid=pid)