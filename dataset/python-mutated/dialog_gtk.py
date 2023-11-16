"""Class for creating Gtk Window dialogs"""
import re
import subprocess
from autokey.scripting.common import DialogData, ColourData

class GtkDialog:
    """
    Provides a simple interface for the display of some basic dialogs to collect information from the user.

    This version uses Zenity to integrate well with GNOME. To pass additional arguments to Zenity that are
    not specifically handled, use keyword arguments. For example, to pass the --timeout argument to Zenity
    pass C{timeout="15"} as one of the parameters. All keyword arguments must be given as strings.

    @note: Exit codes: an exit code of 0 indicates that the user clicked OK.
    """

    def _run_zenity(self, title, args, kwargs) -> DialogData:
        if False:
            return 10
        for (k, v) in kwargs.items():
            args.append('--' + k)
            args.append(v)
        with subprocess.Popen(['zenity', '--title', title] + args, stdout=subprocess.PIPE, universal_newlines=True) as p:
            output = p.communicate()[0][:-1]
            return_code = p.returncode
        return DialogData(return_code, output)

    def info_dialog(self, title='Information', message='', **kwargs):
        if False:
            while True:
                i = 10
        '\n        Show an information dialog\n\n        Usage: C{dialog.info_dialog(title="Information", message="", **kwargs)}\n\n        @param title: window title for the dialog\n        @param message: message displayed in the dialog\n        @return: a tuple containing the exit code and user input\n        @rtype: C{tuple(int, str)}\n        '
        return self._run_zenity(title, ['--info', '--text', message], kwargs)

    def input_dialog(self, title='Enter a value', message='Enter a value', default='', **kwargs):
        if False:
            return 10
        '\n        Show an input dialog\n\n        Usage: C{dialog.input_dialog(title="Enter a value", message="Enter a value", default="", **kwargs)}\n\n        @param title: window title for the dialog\n        @param message: message displayed above the input box\n        @param default: default value for the input box\n        @return: a tuple containing the exit code and user input\n        @rtype: C{DialogData(int, str)}\n        '
        return self._run_zenity(title, ['--entry', '--text', message, '--entry-text', default], kwargs)

    def password_dialog(self, title='Enter password', message='Enter password', **kwargs):
        if False:
            return 10
        '\n        Show a password input dialog\n\n        Usage: C{dialog.password_dialog(title="Enter password", message="Enter password")}\n\n        @param title: window title for the dialog\n        @param message: message displayed above the password input box\n        @return: a tuple containing the exit code and user input\n        @rtype: C{DialogData(int, str)}\n        '
        return self._run_zenity(title, ['--entry', '--text', message, '--hide-text'], kwargs)
        '\n        Show a combobox menu - not supported by zenity\n        \n        Usage: C{dialog.combo_menu(options, title="Choose an option", message="Choose an option")}\n        \n        @param options: list of options (strings) for the dialog\n        @param title: window title for the dialog\n        @param message: message displayed above the combobox      \n        '

    def list_menu(self, options, title='Choose a value', message='Choose a value', default=None, **kwargs):
        if False:
            i = 10
            return i + 15
        '\n        Show a single-selection list menu\n\n        Usage: C{dialog.list_menu(options, title="Choose a value", message="Choose a value", default=None, **kwargs)}\n\n        @param options: list of options (strings) for the dialog\n        @param title: window title for the dialog\n        @param message: message displayed above the list\n        @param default: default value to be selected\n        @return: a tuple containing the exit code and user choice\n        @rtype: C{DialogData(int, str)}\n        '
        choices = []
        for option in options:
            if option == default:
                choices.append('TRUE')
            else:
                choices.append('FALSE')
            choices.append(option)
        return self._run_zenity(title, ['--list', '--radiolist', '--text', message, '--column', ' ', '--column', 'Options'] + choices, kwargs)

    def list_menu_multi(self, options, title='Choose one or more values', message='Choose one or more values', defaults: list=None, **kwargs):
        if False:
            i = 10
            return i + 15
        '\n        Show a multiple-selection list menu\n\n        Usage: C{dialog.list_menu_multi(options, title="Choose one or more values", message="Choose one or more values", defaults=[], **kwargs)}\n\n        @param options: list of options (strings) for the dialog\n        @param title: window title for the dialog\n        @param message: message displayed above the list\n        @param defaults: list of default values to be selected\n        @return: a tuple containing the exit code and user choice\n        @rtype: C{DialogData(int, List[str])}\n        '
        if defaults is None:
            defaults = []
        choices = []
        for option in options:
            if option in defaults:
                choices.append('TRUE')
            else:
                choices.append('FALSE')
            choices.append(option)
        (return_code, output) = self._run_zenity(title, ['--list', '--checklist', '--text', message, '--column', ' ', '--column', 'Options'] + choices, kwargs)
        results = output.split('|')
        return DialogData(return_code, results)

    def open_file(self, title='Open File', **kwargs):
        if False:
            print('Hello World!')
        '\n        Show an Open File dialog\n\n        Usage: C{dialog.open_file(title="Open File", **kwargs)}\n\n        @param title: window title for the dialog\n        @return: a tuple containing the exit code and file path\n        @rtype: C{DialogData(int, str)}\n        '
        return self._run_zenity(title, ['--file-selection'], kwargs)

    def save_file(self, title='Save As', **kwargs):
        if False:
            return 10
        '\n        Show a Save As dialog\n\n        Usage: C{dialog.save_file(title="Save As", **kwargs)}\n\n        @param title: window title for the dialog\n        @return: a tuple containing the exit code and file path\n        @rtype: C{DialogData(int, str)}\n        '
        return self._run_zenity(title, ['--file-selection', '--save'], kwargs)

    def choose_directory(self, title='Select Directory', initialDir='~', **kwargs):
        if False:
            return 10
        '\n        Show a Directory Chooser dialog\n\n        Usage: C{dialog.choose_directory(title="Select Directory", **kwargs)}\n\n        @param title: window title for the dialog\n        @param initialDir:\n        @return: a tuple containing the exit code and path\n        @rtype: C{DialogData(int, str)}\n        '
        return self._run_zenity(title, ['--file-selection', '--directory'], kwargs)

    def choose_colour(self, title='Select Colour', **kwargs):
        if False:
            for i in range(10):
                print('nop')
        '\n        Show a Colour Chooser dialog\n\n        Usage: C{dialog.choose_colour(title="Select Colour")}\n\n        @param title: window title for the dialog\n        @return:\n        @rtype: C{DialogData(int, Optional[ColourData])}\n        '
        return_data = self._run_zenity(title, ['--color-selection'], kwargs)
        if return_data.successful:
            converted_colour = ColourData.from_zenity_tuple_str(return_data.data)
            return DialogData(return_data.return_code, converted_colour)
        else:
            return DialogData(return_data.return_code, None)

    def calendar(self, title='Choose a date', format_str='%Y-%m-%d', date='today', **kwargs):
        if False:
            print('Hello World!')
        '\n        Show a calendar dialog\n\n        Usage: C{dialog.calendar_dialog(title="Choose a date", format="%Y-%m-%d", date="YYYY-MM-DD", **kwargs)}\n\n        @param title: window title for the dialog\n        @param format_str: format of date to be returned\n        @param date: initial date as YYYY-MM-DD, otherwise today\n        @return: a tuple containing the exit code and date\n        @rtype: C{DialogData(int, str)}\n        '
        if re.match('[0-9]{4}-[0-9]{2}-[0-9]{2}', date):
            year = date[0:4]
            month = date[5:7]
            day = date[8:10]
            date_args = ['--year=' + year, '--month=' + month, '--day=' + day]
        else:
            date_args = []
        return self._run_zenity(title, ['--calendar', '--date-format=' + format_str] + date_args, kwargs)