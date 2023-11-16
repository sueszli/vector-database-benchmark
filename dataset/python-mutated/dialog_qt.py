"""Class for creating Qt dialogs"""
import subprocess
from autokey.scripting.common import DialogData, ColourData

class QtDialog:
    """
    Provides a simple interface for the display of some basic dialogs to collect information from the user.

    This version uses KDialog to integrate well with KDE. To pass additional arguments to KDialog that are
    not specifically handled, use keyword arguments. For example, to pass the --geometry argument to KDialog
    to specify the desired size of the dialog, pass C{geometry="700x400"} as one of the parameters. All
    keyword arguments must be given as strings.

    A note on exit codes: an exit code of 0 indicates that the user clicked OK.
    """

    def _run_kdialog(self, title, args, kwargs) -> DialogData:
        if False:
            print('Hello World!')
        for (k, v) in kwargs.items():
            args.append('--' + k)
            args.append(v)
        with subprocess.Popen(['kdialog', '--title', title] + args, stdout=subprocess.PIPE, universal_newlines=True) as p:
            output = p.communicate()[0][:-1]
            return_code = p.returncode
        return DialogData(return_code, output)

    def info_dialog(self, title='Information', message='', **kwargs):
        if False:
            return 10
        '\n        Show an information dialog\n\n        Usage: C{dialog.info_dialog(title="Information", message="", **kwargs)}\n\n        @param title: window title for the dialog\n        @param message: message displayed in the dialog\n        @return: a tuple containing the exit code and user input\n        @rtype: C{DialogData(int, str)}\n        '
        return self._run_kdialog(title, ['--msgbox', message], kwargs)

    def input_dialog(self, title='Enter a value', message='Enter a value', default='', **kwargs):
        if False:
            i = 10
            return i + 15
        '\n        Show an input dialog\n\n        Usage: C{dialog.input_dialog(title="Enter a value", message="Enter a value", default="", **kwargs)}\n\n        @param title: window title for the dialog\n        @param message: message displayed above the input box\n        @param default: default value for the input box\n        @return: a tuple containing the exit code and user input\n        @rtype: C{DialogData(int, str)}\n        '
        return self._run_kdialog(title, ['--inputbox', message, default], kwargs)

    def password_dialog(self, title='Enter password', message='Enter password', **kwargs):
        if False:
            return 10
        '\n        Show a password input dialog\n\n        Usage: C{dialog.password_dialog(title="Enter password", message="Enter password", **kwargs)}\n\n        @param title: window title for the dialog\n        @param message: message displayed above the password input box\n        @return: a tuple containing the exit code and user input\n        @rtype: C{DialogData(int, str)}\n        '
        return self._run_kdialog(title, ['--password', message], kwargs)

    def combo_menu(self, options, title='Choose an option', message='Choose an option', **kwargs):
        if False:
            print('Hello World!')
        '\n        Show a combobox menu\n\n        Usage: C{dialog.combo_menu(options, title="Choose an option", message="Choose an option", **kwargs)}\n\n        @param options: list of options (strings) for the dialog\n        @param title: window title for the dialog\n        @param message: message displayed above the combobox\n        @return: a tuple containing the exit code and user choice\n        @rtype: C{DialogData(int, str)}\n        '
        return self._run_kdialog(title, ['--combobox', message] + options, kwargs)

    def list_menu(self, options, title='Choose a value', message='Choose a value', default=None, **kwargs):
        if False:
            i = 10
            return i + 15
        '\n        Show a single-selection list menu\n\n        Usage: C{dialog.list_menu(options, title="Choose a value", message="Choose a value", default=None, **kwargs)}\n\n        @param options: list of options (strings) for the dialog\n        @param title: window title for the dialog\n        @param message: message displayed above the list\n        @param default: default value to be selected\n        @return: a tuple containing the exit code and user choice\n        @rtype: C{DialogData(int, str)}\n        '
        choices = []
        optionNum = 0
        for option in options:
            choices.append(str(optionNum))
            choices.append(option)
            if option == default:
                choices.append('on')
            else:
                choices.append('off')
            optionNum += 1
        (return_code, result) = self._run_kdialog(title, ['--radiolist', message] + choices, kwargs)
        choice = options[int(result)]
        return DialogData(return_code, choice)

    def list_menu_multi(self, options, title='Choose one or more values', message='Choose one or more values', defaults: list=None, **kwargs):
        if False:
            print('Hello World!')
        '\n        Show a multiple-selection list menu\n\n        Usage: C{dialog.list_menu_multi(options, title="Choose one or more values", message="Choose one or more values", defaults=[], **kwargs)}\n\n        @param options: list of options (strings) for the dialog\n        @param title: window title for the dialog\n        @param message: message displayed above the list\n        @param defaults: list of default values to be selected\n        @return: a tuple containing the exit code and user choice\n        @rtype: C{DialogData(int, List[str])}\n        '
        if defaults is None:
            defaults = []
        choices = []
        optionNum = 0
        for option in options:
            choices.append(str(optionNum))
            choices.append(option)
            if option in defaults:
                choices.append('on')
            else:
                choices.append('off')
            optionNum += 1
        (return_code, output) = self._run_kdialog(title, ['--separate-output', '--checklist', message] + choices, kwargs)
        results = output.split()
        choices = [options[int(choice_index)] for choice_index in results]
        return DialogData(return_code, choices)

    def open_file(self, title='Open File', initialDir='~', fileTypes='*|All Files', rememberAs=None, **kwargs):
        if False:
            i = 10
            return i + 15
        '\n        Show an Open File dialog\n\n        Usage: C{dialog.open_file(title="Open File", initialDir="~", fileTypes="*|All Files", rememberAs=None, **kwargs)}\n\n        @param title: window title for the dialog\n        @param initialDir: starting directory for the file dialog\n        @param fileTypes: file type filter expression\n        @param rememberAs: gives an ID to this file dialog, allowing it to open at the last used path next time\n        @return: a tuple containing the exit code and file path\n        @rtype: C{DialogData(int, str)}\n        '
        if rememberAs is not None:
            return self._run_kdialog(title, ['--getopenfilename', initialDir, fileTypes, ':' + rememberAs], kwargs)
        else:
            return self._run_kdialog(title, ['--getopenfilename', initialDir, fileTypes], kwargs)

    def save_file(self, title='Save As', initialDir='~', fileTypes='*|All Files', rememberAs=None, **kwargs):
        if False:
            return 10
        '\n        Show a Save As dialog\n\n        Usage: C{dialog.save_file(title="Save As", initialDir="~", fileTypes="*|All Files", rememberAs=None, **kwargs)}\n\n        @param title: window title for the dialog\n        @param initialDir: starting directory for the file dialog\n        @param fileTypes: file type filter expression\n        @param rememberAs: gives an ID to this file dialog, allowing it to open at the last used path next time\n        @return: a tuple containing the exit code and file path\n        @rtype: C{DialogData(int, str)}\n        '
        if rememberAs is not None:
            return self._run_kdialog(title, ['--getsavefilename', initialDir, fileTypes, ':' + rememberAs], kwargs)
        else:
            return self._run_kdialog(title, ['--getsavefilename', initialDir, fileTypes], kwargs)

    def choose_directory(self, title='Select Directory', initialDir='~', rememberAs=None, **kwargs):
        if False:
            print('Hello World!')
        '\n        Show a Directory Chooser dialog\n\n        Usage: C{dialog.choose_directory(title="Select Directory", initialDir="~", rememberAs=None, **kwargs)}\n\n        @param title: window title for the dialog\n        @param initialDir: starting directory for the directory chooser dialog\n        @param rememberAs: gives an ID to this file dialog, allowing it to open at the last used path next time\n        @return: a tuple containing the exit code and chosen path\n        @rtype: C{DialogData(int, str)}\n        '
        if rememberAs is not None:
            return self._run_kdialog(title, ['--getexistingdirectory', initialDir, ':' + rememberAs], kwargs)
        else:
            return self._run_kdialog(title, ['--getexistingdirectory', initialDir], kwargs)

    def choose_colour(self, title='Select Colour', **kwargs):
        if False:
            while True:
                i = 10
        '\n        Show a Colour Chooser dialog\n\n        Usage: C{dialog.choose_colour(title="Select Colour")}\n\n        @param title: window title for the dialog\n        @return: a tuple containing the exit code and colour\n        @rtype: C{DialogData(int, str)}\n        '
        return_data = self._run_kdialog(title, ['--getcolor'], kwargs)
        if return_data.successful:
            return DialogData(return_data.return_code, ColourData.from_html(return_data.data))
        else:
            return DialogData(return_data.return_code, None)

    def calendar(self, title='Choose a date', format_str='%Y-%m-%d', date='today', **kwargs):
        if False:
            while True:
                i = 10
        '\n        Show a calendar dialog\n\n        Usage: C{dialog.calendar_dialog(title="Choose a date", format="%Y-%m-%d", date="YYYY-MM-DD", **kwargs)}\n\n        Note: the format and date parameters are not currently used\n\n        @param title: window title for the dialog\n        @param format_str: format of date to be returned\n        @param date: initial date as YYYY-MM-DD, otherwise today\n        @return: a tuple containing the exit code and date\n        @rtype: C{DialogData(int, str)}\n        '
        return self._run_kdialog(title, ['--calendar', title], kwargs)