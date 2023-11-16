import os
import sys
import ryven.main.packages.nodes_package
from ryven.main import utils
from ryven.main.config import Config
from ryven.main.args_parser import process_args

def run(*args_, qt_app=None, gui_parent=None, use_sysargs=True, **kwargs):
    if False:
        while True:
            i = 10
    "Start the Ryven window.\n\n    The `*args_` and `**kwargs` arguments correspond to their positional and\n    optional command line equivalents, respectively (see `parse_args()`).\n    Optional keyword arguments are specified without the leading double hyphens\n    '--'. As a name, the corresponding 'dest' value of `add_argument` has to\n    be used. E.g. the command line:\n        ryven --window-theme=light --nodes=std --nodes=linalg myproject.json\n    becomes:\n        run('myproject.json', window_theme='light', nodes=['std', 'linalg'])\n\n    Note 1\n    ------\n    The `*args_` and `**kwargs` takes predecence and overwrites the values\n    specified on the command line (or the default values, if\n    `use_sysargs=False` is given). The exception are lists, which are appended,\n    e.g. in the example from above, the two nodes 'std' and 'linalg' are added\n    at the end of the nodes supplied at the command line.\n\n    Note 2\n    ------\n    The positional command line argument to specify the project file also\n    checks `utils.ryven_dir_path()/saves`, if it can find the project file.\n    The `*args_` does not perform this check. This is up to the developer\n    calling `run()`. The developer can always use `utils.find_project()` to\n    find projects in this additional directory.\n\n    Parameters\n    ----------\n    qt_app : QApplication, optional\n        The `QApplication` to be used. If `None` a `QApplication` is generated.\n        The default is `None`.\n    gui_parent : QWidget, optional\n        The parent `QWidget`.\n        The default is `None`.\n    use_sysargs : bool, optional\n        Whether the command line arguments should be used.\n        The default is `True`.\n    *args_ : str\n        Corresponding to the positional command line argument(s).\n    **kwargs : any\n        Corresponding to the keyword command line arguments.\n\n    Raises\n    ------\n    TypeError\n        Raised, if keyword argument is not specified by the argument parser or\n        the wrong number of positional arguments are specified.\n\n    Returns\n    -------\n    None|Main Window\n    "
    conf: Config = process_args(use_sysargs, *args_, **kwargs)
    os.environ['RYVEN_MODE'] = 'gui'
    os.environ['QT_API'] = conf.qt_api
    from ryven.node_env import init_node_env
    from ryven.gui_env import init_node_guis_env
    init_node_env()
    init_node_guis_env()
    from ryven.gui.main_console import init_main_console
    from ryven.gui.main_window import MainWindow
    from ryven.gui.styling.window_theme import apply_stylesheet
    if qt_app is None:
        from qtpy.QtWidgets import QApplication
        if conf.window_geometry:
            qt_args = [sys.argv[0], '-geometry', conf.window_geometry]
        else:
            qt_args = [sys.argv[0]]
        app = QApplication(qt_args)
    else:
        app = qt_app
    from qtpy.QtGui import QFontDatabase
    db = QFontDatabase()
    db.addApplicationFont(utils.abs_path_from_package_dir('resources/fonts/poppins/Poppins-Medium.ttf'))
    db.addApplicationFont(utils.abs_path_from_package_dir('resources/fonts/source_code_pro/SourceCodePro-Regular.ttf'))
    db.addApplicationFont(utils.abs_path_from_package_dir('resources/fonts/asap/Asap-Regular.ttf'))
    if conf.show_dialog:
        from ryven.gui.startup_dialog.StartupDialog import StartupDialog
        sw = StartupDialog(config=conf, parent=gui_parent)
        if sw.exec_() <= 0:
            sys.exit('Start-up screen dismissed')
    if conf.nodes:
        (conf.nodes, pkgs_not_found, _) = ryven.main.packages.nodes_package.process_nodes_packages(list(conf.nodes))
        if pkgs_not_found:
            sys.exit(f"Error: Nodes packages not found: {', '.join([str(p) for p in pkgs_not_found])}")
    conf.window_theme = apply_stylesheet(conf.window_theme)
    if conf.flow_theme is None:
        if conf.window_theme.name == 'dark':
            conf.flow_theme = 'pure dark'
        else:
            conf.flow_theme = 'pure light'
    if conf.project:
        (pkgs, pkgs_not_found, project_dict) = ryven.main.packages.nodes_package.process_nodes_packages(conf.project, requested_packages=list(conf.nodes))
        if pkgs_not_found:
            str_missing_pkgs = ', '.join([str(p.name) for p in pkgs_not_found])
            plural = len(pkgs_not_found) > 1
            sys.exit(f"""The package{('s' if plural else '')} {str_missing_pkgs}{('were' if plural else 'was')} requested, but {('they are' if plural else 'it is')} not available.\nUpdate the project file or supply the missing package{('s' if plural else '')} {str_missing_pkgs} on the command line with the "-n" switch.""")
        requested_packages = conf.nodes
        required_packages = pkgs
        project_content = project_dict
    else:
        requested_packages = conf.nodes
        required_packages = None
        project_content = None
    (console_stdout_redirect, console_errout_redirect) = init_main_console(conf.window_theme)
    editor = MainWindow(config=conf, requested_packages=requested_packages, required_packages=required_packages, project_content=project_content, parent=gui_parent)
    editor.show()
    if qt_app is None:
        if conf.verbose:
            editor.print_info()
            sys.exit(app.exec_())
        else:
            import contextlib
            with contextlib.redirect_stdout(console_stdout_redirect), contextlib.redirect_stderr(console_errout_redirect):
                editor.print_info()
                sys.exit(app.exec_())
    else:
        return editor
if __name__ == '__main__':
    run()