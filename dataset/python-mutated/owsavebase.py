import os.path
import sys
import re
from AnyQt.QtWidgets import QFileDialog, QGridLayout, QMessageBox
from Orange.widgets import gui, widget
from Orange.widgets.settings import Setting
_userhome = os.path.expanduser(f'~{os.sep}')

class OWSaveBase(widget.OWWidget, openclass=True):
    """
    Base class for Save widgets

    A derived class must provide, at minimum:

    - class `Inputs` and the corresponding handler that:

      - saves the input to an attribute `data`, and
      - calls `self.on_new_input`.

    - a class attribute `filters` with a list of filters or a dictionary whose
      keys are filters OR a class method `get_filters` that returns such a
      list or dictionary
    - method `do_save` that saves `self.data` into `self.filename`

    Alternatively, instead of defining `do_save` a derived class can make
    `filters` a dictionary whose keys are classes that define a method `write`
    (like e.g. `TabReader`). Method `do_save` defined in the base class calls
    the writer corresponding to the currently chosen filter.

    A minimum example of derived class is
    `Orange.widgets.model.owsavemodel.OWSaveModel`.
    A more advanced widget that overrides a lot of base class behaviour is
    `Orange.widgets.data.owsave.OWSave`.
    """

    class Information(widget.OWWidget.Information):
        empty_input = widget.Msg('Empty input; nothing was saved.')

    class Warning(widget.OWWidget.Warning):
        auto_save_disabled = widget.Msg('Auto save disabled.\nDue to security reasons auto save is only restored for paths that are in the same directory as the workflow file or in a subtree of that directory.')

    class Error(widget.OWWidget.Error):
        no_file_name = widget.Msg('File name is not set.')
        unsupported_format = widget.Msg('File format is unsupported.\n{}')
        general_error = widget.Msg('{}')
    want_main_area = False
    resizing_enabled = False
    filter = Setting('')
    stored_path = Setting('')
    stored_name = Setting('', schema_only=True)
    auto_save = Setting(False, schema_only=True)
    filters = []

    def __init__(self, start_row=0):
        if False:
            print('Hello World!')
        '\n        Set up the gui.\n\n        The gui consists of a checkbox for auto save and two buttons put on a\n        grid layout. Derived widgets that want to place controls above the auto\n        save widget can set the `start_row` argument to the first free row,\n        and this constructor will start filling the grid there.\n\n        Args:\n            start_row (int): the row at which to start filling the gui\n        '
        super().__init__()
        self.data = None
        self.__show_auto_save_disabled = False
        self._absolute_path = self._abs_path_from_setting()
        if not self.filter:
            self.filter = self.default_filter()
        self.grid = grid = QGridLayout()
        gui.widgetBox(self.controlArea, orientation=grid, box=True)
        grid.addWidget(gui.checkBox(None, self, 'auto_save', 'Autosave when receiving new data', callback=self._on_auto_save_toggled), start_row, 0, 1, 2)
        self.bt_save = gui.button(self.buttonsArea, self, label=f'Save as {self.stored_name}' if self.stored_name else 'Save', callback=self.save_file)
        gui.button(self.buttonsArea, self, 'Save as ...', callback=self.save_file_as)
        self.adjustSize()
        self.update_messages()

    def default_filter(self):
        if False:
            print('Hello World!')
        'Returns the first filter in the list'
        return next(iter(self.get_filters()))

    @property
    def last_dir(self):
        if False:
            for i in range(10):
                print('nop')
        return self._absolute_path

    @last_dir.setter
    def last_dir(self, absolute_path):
        if False:
            while True:
                i = 10
        'Store _absolute_path and update relative path (stored_path)'
        self._absolute_path = absolute_path
        self.stored_path = absolute_path
        workflow_dir = self.workflowEnv().get('basedir', None)
        if workflow_dir:
            relative_path = os.path.relpath(absolute_path, start=workflow_dir)
            if not relative_path.startswith('..'):
                self.stored_path = relative_path

    def _abs_path_from_setting(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Compute absolute path from `stored_path` from settings.\n\n        Absolute stored path is used only if it exists.\n        Auto save is disabled unless stored_path is relative.\n        '
        workflow_dir = self.workflowEnv().get('basedir')
        if os.path.isabs(self.stored_path):
            if os.path.exists(self.stored_path):
                self._disable_auto_save_and_warn()
                return self.stored_path
        elif workflow_dir is not None:
            return os.path.normpath(os.path.join(workflow_dir, self.stored_path))
        self.stored_path = workflow_dir or _userhome
        self.auto_save = False
        return self.stored_path

    def _disable_auto_save_and_warn(self):
        if False:
            print('Hello World!')
        if self.auto_save:
            self.__show_auto_save_disabled = True
        self.auto_save = False

    def _on_auto_save_toggled(self):
        if False:
            while True:
                i = 10
        self.__show_auto_save_disabled = False
        self.update_messages()

    @property
    def filename(self):
        if False:
            i = 10
            return i + 15
        if self.stored_name:
            return os.path.join(self._absolute_path, self.stored_name)
        else:
            return ''

    @filename.setter
    def filename(self, value):
        if False:
            for i in range(10):
                print('nop')
        (self.last_dir, self.stored_name) = os.path.split(value)

    def workflowEnvChanged(self, key, value, oldvalue):
        if False:
            while True:
                i = 10
        if key == 'basedir':
            self.last_dir = self._absolute_path

    @classmethod
    def get_filters(cls):
        if False:
            i = 10
            return i + 15
        return cls.filters

    @property
    def writer(self):
        if False:
            while True:
                i = 10
        '\n        Return the active writer or None if there is no writer for this filter\n\n        The base class uses this property only in `do_save` to find the writer\n        corresponding to the filter. Derived classes (e.g. OWSave) may also use\n        it elsewhere.\n\n        Filter may not exist if it comes from settings saved in Orange with\n        some add-ons that are not (or no longer) present, or if support for\n        some extension was dropped, like the old Excel format.\n        '
        filters = self.get_filters()
        if self.filter not in filters:
            return None
        return filters[self.filter]

    def on_new_input(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        This method must be called from input signal handler.\n\n        - It clears errors, warnings and information and calls\n          `self.update_messages` to set the as needed.\n        - It also calls `update_status` the can be overriden in derived\n          methods to set the status (e.g. the number of input rows)\n        - Calls `self.save_file` if `self.auto_save` is enabled and\n          `self.filename` is provided.\n        '
        self.Error.clear()
        self.Warning.clear()
        self.Information.clear()
        self.update_messages()
        self.update_status()
        if self.auto_save and self.filename:
            self.save_file()

    def save_file_as(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Ask the user for the filename and try saving the file\n        '
        (filename, selected_filter) = self.get_save_filename()
        if not filename:
            return
        self.filename = filename
        self.filter = selected_filter
        self.Error.unsupported_format.clear()
        self.bt_save.setText(f'Save as {self.stored_name}')
        self.update_messages()
        self._try_save()

    def save_file(self):
        if False:
            i = 10
            return i + 15
        '\n        If file name is provided, try saving, else call save_file_as\n        '
        if not self.filename:
            self.save_file_as()
        else:
            self._try_save()

    def _try_save(self):
        if False:
            print('Hello World!')
        '\n        Private method that calls do_save within try-except that catches and\n        shows IOError. Do nothing if not data or no file name.\n        '
        self.Error.general_error.clear()
        if self.data is None or not self.filename:
            return
        try:
            self.do_save()
        except IOError as err_value:
            self.Error.general_error(str(err_value))

    def do_save(self):
        if False:
            return 10
        '\n        Do the saving.\n\n        Default implementation calls the write method of the writer\n        corresponding to the current filter. This requires that get_filters()\n        returns is a dictionary whose keys are classes.\n\n        Derived classes may simplify this by providing a list of filters and\n        override do_save. This is particularly handy if the widget supports only\n        a single format.\n        '
        if self.writer is None:
            self.Error.unsupported_format(self.filter)
            return
        self.writer.write(self.filename, self.data)

    def update_messages(self):
        if False:
            print('Hello World!')
        '\n        Update errors, warnings and information.\n\n        Default method sets no_file_name if auto_save is enabled but file name\n        is not provided; and empty_input if file name is given but there is no\n        data.\n\n        Derived classes that define further messages will typically set them in\n        this method.\n        '
        self.Error.no_file_name(shown=not self.filename and self.auto_save)
        self.Information.empty_input(shown=self.filename and self.data is None)
        self.Warning.auto_save_disabled(shown=self.__show_auto_save_disabled)

    def update_status(self):
        if False:
            print('Hello World!')
        '\n        Update the input/output indicator. Default method does nothing.\n        '

    def initial_start_dir(self):
        if False:
            print('Hello World!')
        "\n        Provide initial start directory\n\n        Return either the current file's path, the last directory or home.\n        "
        if self.filename and os.path.exists(os.path.split(self.filename)[0]):
            return self.filename
        else:
            return self.last_dir or _userhome

    @staticmethod
    def suggested_name():
        if False:
            i = 10
            return i + 15
        '\n        Suggest the name for the output file or return an empty string.\n        '
        return ''

    @classmethod
    def _replace_extension(cls, filename, extension):
        if False:
            print('Hello World!')
        '\n        Remove all extensions that appear in any filter.\n\n        Double extensions are broken in different weird ways across all systems,\n        including omitting some, like turning iris.tab.gz to iris.gz. This\n        function removes anything that can appear anywhere.\n        '
        known_extensions = set()
        for filt in cls.get_filters():
            known_extensions |= set(cls._extension_from_filter(filt).split('.'))
        if '' in known_extensions:
            known_extensions.remove('')
        while True:
            (base, ext) = os.path.splitext(filename)
            if ext[1:] not in known_extensions:
                break
            filename = base
        return filename + extension

    @staticmethod
    def _extension_from_filter(selected_filter):
        if False:
            while True:
                i = 10
        return re.search('.*\\(\\*?(\\..*)\\)$', selected_filter).group(1)

    def valid_filters(self):
        if False:
            print('Hello World!')
        return self.get_filters()

    def default_valid_filter(self):
        if False:
            for i in range(10):
                print('nop')
        return self.filter

    @classmethod
    def migrate_settings(cls, settings, version):
        if False:
            i = 10
            return i + 15
        if 'last_dir' in settings:
            settings['stored_path'] = settings.pop('last_dir')
        if 'filename' in settings:
            settings['stored_name'] = os.path.split(settings.pop('filename') or '')[1]
    if sys.platform in ('darwin', 'win32'):

        def get_save_filename(self):
            if False:
                while True:
                    i = 10
            if sys.platform == 'darwin':

                def remove_star(filt):
                    if False:
                        i = 10
                        return i + 15
                    return filt.replace(' (*.', ' (.')
            else:

                def remove_star(filt):
                    if False:
                        return 10
                    return filt
            no_ext_filters = {remove_star(f): f for f in self.valid_filters()}
            filename = self.initial_start_dir()
            while True:
                dlg = QFileDialog(None, 'Save File', filename, ';;'.join(no_ext_filters))
                dlg.setAcceptMode(dlg.AcceptSave)
                dlg.selectNameFilter(remove_star(self.default_valid_filter()))
                dlg.setOption(QFileDialog.DontConfirmOverwrite)
                if dlg.exec() == QFileDialog.Rejected:
                    return ('', '')
                filename = dlg.selectedFiles()[0]
                selected_filter = no_ext_filters[dlg.selectedNameFilter()]
                filename = self._replace_extension(filename, self._extension_from_filter(selected_filter))
                if not os.path.exists(filename) or QMessageBox.question(self, 'Overwrite file?', f'File {os.path.split(filename)[1]} already exists.\nOverwrite?') == QMessageBox.Yes:
                    return (filename, selected_filter)
    else:

        class SaveFileDialog(QFileDialog):

            def __init__(self, save_cls, *args, **kwargs):
                if False:
                    for i in range(10):
                        print('nop')
                super().__init__(*args, **kwargs)
                self.save_cls = save_cls
                self.suffix = ''
                self.setAcceptMode(QFileDialog.AcceptSave)
                self.setOption(QFileDialog.DontUseNativeDialog)
                self.filterSelected.connect(self.updateDefaultExtension)

            def selectNameFilter(self, selected_filter):
                if False:
                    print('Hello World!')
                super().selectNameFilter(selected_filter)
                self.updateDefaultExtension(selected_filter)

            def updateDefaultExtension(self, selected_filter):
                if False:
                    while True:
                        i = 10
                self.suffix = self.save_cls._extension_from_filter(selected_filter)
                files = self.selectedFiles()
                if files and (not os.path.isdir(files[0])):
                    self.selectFile(files[0])

            def selectFile(self, filename):
                if False:
                    while True:
                        i = 10
                filename = self.save_cls._replace_extension(filename, self.suffix)
                super().selectFile(filename)

        def get_save_filename(self):
            if False:
                while True:
                    i = 10
            dlg = self.SaveFileDialog(type(self), None, 'Save File', self.initial_start_dir(), ';;'.join(self.valid_filters()))
            dlg.selectNameFilter(self.default_valid_filter())
            if dlg.exec() == QFileDialog.Rejected:
                return ('', '')
            else:
                return (dlg.selectedFiles()[0], dlg.selectedNameFilter())