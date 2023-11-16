import os.path
import subprocess
import sys
import mmap, re
import warnings
import PySimpleGUI as sg
'\n    PySimpleGUI Demo Program Browser\n\n    Originaly written for PySimpleGUI Demo Programs, but expanded to\n    be a general purpose tool. Enable Advanced Mode in settings for more fun\n    \n    Use to filter and search your source code tree.\n        Then run or edit your files\n\n    Filter the list of :\n        * Search using filename\n        * Searching within the programs\' source code (like grep)\n    \n    The basic file operations are\n        * Edit a file in your editor\n        * Run a file\n        * Filter file list\n        * Search in files\n        * Run a regular expression search on all files\n        * Display the matching line in a file\n    \n    Additional operations\n        * Edit this file in editor\n        \n    Keeps a "history" of the previously chosen folders to easy switching between projects\n                \n    Copyright 2021 PySimpleGUI.org\n'

def running_linux():
    if False:
        while True:
            i = 10
    return sys.platform.startswith('linux')

def running_windows():
    if False:
        i = 10
        return i + 15
    return sys.platform.startswith('win')

def get_file_list_dict():
    if False:
        print('Hello World!')
    '\n    Returns dictionary of files\n    Key is short filename\n    Value is the full filename and path\n\n    :return: Dictionary of demo files\n    :rtype: Dict[str:str]\n    '
    demo_path = get_demo_path()
    demo_files_dict = {}
    for (dirname, dirnames, filenames) in os.walk(demo_path):
        for filename in filenames:
            if filename.endswith('.py') or filename.endswith('.pyw'):
                fname_full = os.path.join(dirname, filename)
                if filename not in demo_files_dict.keys():
                    demo_files_dict[filename] = fname_full
                else:
                    for i in range(1, 100):
                        new_filename = f'{filename}_{i}'
                        if new_filename not in demo_files_dict:
                            demo_files_dict[new_filename] = fname_full
                            break
    return demo_files_dict

def get_file_list():
    if False:
        i = 10
        return i + 15
    '\n    Returns list of filenames of files to display\n    No path is shown, only the short filename\n\n    :return: List of filenames\n    :rtype: List[str]\n    '
    return sorted(list(get_file_list_dict().keys()))

def get_demo_path():
    if False:
        i = 10
        return i + 15
    '\n    Get the top-level folder path\n    :return: Path to list of files using the user settings for this file.  Returns folder of this file if not found\n    :rtype: str\n    '
    demo_path = sg.user_settings_get_entry('-demos folder-', os.path.dirname(__file__))
    return demo_path

def get_global_editor():
    if False:
        print('Hello World!')
    "\n    Get the path to the editor based on user settings or on PySimpleGUI's global settings\n\n    :return: Path to the editor\n    :rtype: str\n    "
    try:
        global_editor = sg.pysimplegui_user_settings.get('-editor program-')
    except:
        global_editor = ''
    return global_editor

def get_editor():
    if False:
        return 10
    "\n    Get the path to the editor based on user settings or on PySimpleGUI's global settings\n\n    :return: Path to the editor\n    :rtype: str\n    "
    try:
        global_editor = sg.pysimplegui_user_settings.get('-editor program-')
    except:
        global_editor = ''
    user_editor = sg.user_settings_get_entry('-editor program-', '')
    if user_editor == '':
        user_editor = global_editor
    return user_editor

def get_explorer():
    if False:
        i = 10
        return i + 15
    '\n    Get the path to the file explorer program\n\n    :return: Path to the file explorer EXE\n    :rtype: str\n    '
    try:
        global_explorer = sg.pysimplegui_user_settings.get('-explorer program-', '')
    except:
        global_explorer = ''
    explorer = sg.user_settings_get_entry('-explorer program-', '')
    if explorer == '':
        explorer = global_explorer
    return explorer

def advanced_mode():
    if False:
        i = 10
        return i + 15
    '\n    Returns True is advanced GUI should be shown\n\n    :return: True if user indicated wants the advanced GUI to be shown (set in the settings window)\n    :rtype: bool\n    '
    return sg.user_settings_get_entry('-advanced mode-', False)

def get_theme():
    if False:
        while True:
            i = 10
    "\n    Get the theme to use for the program\n    Value is in this program's user settings. If none set, then use PySimpleGUI's global default theme\n    :return: The theme\n    :rtype: str\n    "
    try:
        global_theme = sg.theme_global()
    except:
        global_theme = sg.theme()
    user_theme = sg.user_settings_get_entry('-theme-', '')
    if user_theme == '':
        user_theme = global_theme
    return user_theme
warnings.filterwarnings('ignore', category=DeprecationWarning)

def get_line_number(file_path, string):
    if False:
        return 10
    lmn = 0
    with open(file_path) as f:
        for (num, line) in enumerate(f, 1):
            if string.strip() == line.strip():
                lmn = num
    return lmn

def find_in_file(string, demo_files_dict, regex=False, verbose=False, window=None, ignore_case=True, show_first_match=True):
    if False:
        return 10
    '\n    Search through the demo files for a string.\n    The case of the string and the file contents are ignored\n\n    :param string: String to search for\n    :param verbose: if True print the FIRST match\n    :type verbose: bool\n    :param find_all_matches: if True, then return all matches in the dictionary\n    :type find_all_matches: bool\n    :return: List of files containing the string\n    :rtype: List[str]\n    '
    file_list = []
    num_files = 0
    matched_dict = {}
    for file in demo_files_dict:
        try:
            full_filename = demo_files_dict[file]
            if not demo_files_dict == get_file_list_dict():
                full_filename = full_filename[0]
            matches = None
            with open(full_filename, 'rb', 0) as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as s:
                if regex:
                    window['-FIND NUMBER-'].update(f'{num_files} files')
                    window.refresh()
                    matches = re.finditer(bytes('^.*(' + string + ').*$', 'utf-8'), s, re.MULTILINE)
                    if matches:
                        for match in matches:
                            if match is not None:
                                if file not in file_list:
                                    file_list.append(file)
                                    num_files += 1
                                if verbose:
                                    sg.cprint(f'{file}:', c='white on green')
                                    sg.cprint(f"{match.group(0).decode('utf-8')}\n")
                else:
                    window['-FIND NUMBER-'].update(f'{num_files} files')
                    window.refresh()
                    matches = None
                    if ignore_case:
                        if show_first_match:
                            matches = re.search(b'(?i)^' + bytes('.*(' + re.escape(string.lower()) + ').*$', 'utf-8'), s, re.MULTILINE)
                        else:
                            matches = re.finditer(b'(?i)^' + bytes('.*(' + re.escape(string.lower()) + ').*$', 'utf-8'), s, re.MULTILINE)
                    elif show_first_match:
                        matches = re.search(b'^' + bytes('.*(' + re.escape(string) + ').*$', 'utf-8'), s, re.MULTILINE)
                    else:
                        matches = re.finditer(b'^' + bytes('.*(' + re.escape(string) + ').*$', 'utf-8'), s, re.MULTILINE)
                    if matches:
                        if show_first_match:
                            file_list.append(file)
                            num_files += 1
                            match_array = []
                            match_array.append(matches.group(0).decode('utf-8'))
                            matched_dict[full_filename] = match_array
                        else:
                            append_file = False
                            match_array = []
                            for match_ in matches:
                                match_str = match_.group(0).decode('utf-8')
                                if match_str:
                                    match_array.append(match_str)
                                    if append_file == False:
                                        append_file = True
                            if append_file:
                                file_list.append(file)
                                num_files += 1
                                matched_dict[full_filename] = match_array
        except ValueError:
            del matches
        except Exception as e:
            (exc_type, exc_obj, exc_tb) = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)
            print(f'{file}', e, file=sys.stderr)
    file_lines_dict = {}
    if not regex:
        for key in matched_dict:
            (head, tail) = os.path.split(key)
            file_array_old = [key]
            file_array_new = []
            if verbose:
                sg.cprint(f'{tail}:', c='white on green')
            try:
                for _match in matched_dict[key]:
                    line_num_match = get_line_number(key, _match)
                    file_array_new.append(line_num_match)
                    if verbose:
                        sg.cprint(f'Line: {line_num_match} | {_match.strip()}\n')
                file_array_old.append(file_array_new)
                file_lines_dict[tail] = file_array_old
            except:
                pass
        find_in_file.old_file_list = file_lines_dict
    file_list = list(set(file_list))
    return file_list

def settings_window():
    if False:
        while True:
            i = 10
    '\n    Show the settings window.\n    This is where the folder paths and program paths are set.\n    Returns True if settings were changed\n\n    :return: True if settings were changed\n    :rtype: (bool)\n    '
    try:
        global_editor = sg.pysimplegui_user_settings.get('-editor program-')
    except:
        global_editor = ''
    try:
        global_explorer = sg.pysimplegui_user_settings.get('-explorer program-')
    except:
        global_explorer = ''
    try:
        global_theme = sg.theme_global()
    except:
        global_theme = ''
    layout = [[sg.T('Program Settings', font='DEFAULT 25')], [sg.T('Path to Tree', font='_ 16')], [sg.Combo(sorted(sg.user_settings_get_entry('-folder names-', [])), default_value=sg.user_settings_get_entry('-demos folder-', get_demo_path()), size=(50, 1), key='-FOLDERNAME-'), sg.FolderBrowse('Folder Browse', target='-FOLDERNAME-'), sg.B('Clear History')], [sg.T('Editor Program', font='_ 16')], [sg.T('Leave blank to use global default'), sg.T(global_editor)], [sg.In(sg.user_settings_get_entry('-editor program-', ''), k='-EDITOR PROGRAM-'), sg.FileBrowse()], [sg.T('File Explorer Program', font='_ 16')], [sg.T('Leave blank to use global default'), sg.T(global_explorer)], [sg.In(sg.user_settings_get_entry('-explorer program-'), k='-EXPLORER PROGRAM-'), sg.FileBrowse()], [sg.T('Theme', font='_ 16')], [sg.T('Leave blank to use global default'), sg.T(global_theme)], [sg.Combo([''] + sg.theme_list(), sg.user_settings_get_entry('-theme-', ''), readonly=True, k='-THEME-')], [sg.CB('Use Advanced Interface', default=advanced_mode(), k='-ADVANCED MODE-')], [sg.B('Ok', bind_return_key=True), sg.B('Cancel')]]
    window = sg.Window('Settings', layout)
    settings_changed = False
    while True:
        (event, values) = window.read()
        if event in ('Cancel', sg.WIN_CLOSED):
            break
        if event == 'Ok':
            sg.user_settings_set_entry('-demos folder-', values['-FOLDERNAME-'])
            sg.user_settings_set_entry('-editor program-', values['-EDITOR PROGRAM-'])
            sg.user_settings_set_entry('-theme-', values['-THEME-'])
            sg.user_settings_set_entry('-folder names-', list(set(sg.user_settings_get_entry('-folder names-', []) + [values['-FOLDERNAME-']])))
            sg.user_settings_set_entry('-explorer program-', values['-EXPLORER PROGRAM-'])
            sg.user_settings_set_entry('-advanced mode-', values['-ADVANCED MODE-'])
            settings_changed = True
            break
        elif event == 'Clear History':
            sg.user_settings_set_entry('-folder names-', [])
            sg.user_settings_set_entry('-last filename-', '')
            window['-FOLDERNAME-'].update(values=[], value='')
    window.close()
    return settings_changed
ML_KEY = '-ML-'

def make_window():
    if False:
        i = 10
        return i + 15
    '\n    Creates the main window\n    :return: The main window object\n    :rtype: (Window)\n    '
    theme = get_theme()
    if not theme:
        theme = sg.OFFICIAL_PYSIMPLEGUI_THEME
    sg.theme(theme)
    find_tooltip = 'Find in file\nEnter a string in box to search for string inside of the files.\nFile list will update with list of files string found inside.'
    filter_tooltip = 'Filter files\nEnter a string in box to narrow down the list of files.\nFile list will update with list of files with string in filename.'
    find_re_tooltip = 'Find in file using Regular Expression\nEnter a string in box to search for string inside of the files.\nSearch is performed after clicking the FindRE button.'
    left_col = sg.Col([[sg.Listbox(values=get_file_list(), select_mode=sg.SELECT_MODE_EXTENDED, size=(50, 20), key='-DEMO LIST-')], [sg.Text('Filter:', tooltip=filter_tooltip), sg.Input(size=(25, 1), enable_events=True, key='-FILTER-', tooltip=filter_tooltip), sg.T(size=(15, 1), k='-FILTER NUMBER-')], [sg.Button('Run'), sg.B('Edit'), sg.B('Clear'), sg.B('Open Folder')], [sg.Text('Find:', tooltip=find_tooltip), sg.Input(size=(25, 1), enable_events=True, key='-FIND-', tooltip=find_tooltip), sg.T(size=(15, 1), k='-FIND NUMBER-')]], element_justification='l')
    lef_col_find_re = sg.pin(sg.Col([[sg.Text('Find:', tooltip=find_re_tooltip), sg.Input(size=(25, 1), key='-FIND RE-', tooltip=find_re_tooltip), sg.B('Find RE')]], k='-RE COL-'))
    right_col = [[sg.Multiline(size=(70, 21), write_only=True, key=ML_KEY, reroute_stdout=True, echo_stdout_stderr=True)], [sg.Button('Edit Me (this program)'), sg.B('Settings'), sg.Button('Exit')], [sg.T('PySimpleGUI ver ' + sg.version.split(' ')[0] + '  tkinter ver ' + sg.tclversion_detailed, font='Default 8', pad=(0, 0))]]
    options_at_bottom = sg.pin(sg.Column([[sg.CB('Verbose', enable_events=True, k='-VERBOSE-'), sg.CB('Show only first match in file', default=True, enable_events=True, k='-FIRST MATCH ONLY-'), sg.CB('Find ignore case', default=True, enable_events=True, k='-IGNORE CASE-'), sg.T('<---- NOTE: Only the "Verbose" setting is implemented at this time')]], pad=(0, 0), k='-OPTIONS BOTTOM-'))
    choose_folder_at_top = sg.pin(sg.Column([[sg.T('Click settings to set top of your tree or choose a previously chosen folder'), sg.Combo(sorted(sg.user_settings_get_entry('-folder names-', [])), default_value=sg.user_settings_get_entry('-demos folder-', ''), size=(50, 1), key='-FOLDERNAME-', enable_events=True, readonly=True)]], pad=(0, 0), k='-FOLDER CHOOSE-'))
    layout = [[sg.Text('PySimpleGUI Demo Program & Project Browser', font='Any 20')], [choose_folder_at_top], sg.vtop([sg.Column([[left_col], [lef_col_find_re]], element_justification='l'), sg.Col(right_col, element_justification='c')]), [options_at_bottom]]
    window = sg.Window('PSG Demo & Project Browser', layout, finalize=True, icon=icon)
    if not advanced_mode():
        window['-FOLDER CHOOSE-'].update(visible=False)
        window['-RE COL-'].update(visible=False)
        window['-OPTIONS BOTTOM-'].update(visible=False)
    sg.cprint_set_output_destination(window, ML_KEY)
    return window

def main():
    if False:
        while True:
            i = 10
    '\n    The main program that contains the event loop.\n    It will call the make_window function to create the window.\n    '
    find_in_file.old_file_list = None
    old_typed_value = None
    file_list_dict = get_file_list_dict()
    file_list = get_file_list()
    window = make_window()
    window['-FILTER NUMBER-'].update(f'{len(file_list)} files')
    counter = 0
    while True:
        (event, values) = window.read()
        counter += 1
        if event in (sg.WINDOW_CLOSED, 'Exit'):
            break
        if event == 'Edit':
            editor_program = get_editor()
            for file in values['-DEMO LIST-']:
                sg.cprint(f'Editing using {editor_program}', text_color='white', background_color='red', end='')
                sg.cprint('')
                sg.cprint(f'{file_list_dict[file]}', text_color='white', background_color='purple')
                execute_command_subprocess(f'{editor_program}', f'"{file_list_dict[file]}"')
        elif event == 'Run':
            sg.cprint('Running....', c='white on green', end='')
            sg.cprint('')
            for file in values['-DEMO LIST-']:
                file_to_run = str(file_list_dict[file])
                sg.cprint(file_to_run, text_color='white', background_color='purple')
                execute_py_file(f'{file_to_run}')
        elif event.startswith('Edit Me'):
            editor_program = get_editor()
            sg.cprint(f'opening using {editor_program}:')
            sg.cprint(f'{__file__}', text_color='white', background_color='red', end='')
            execute_command_subprocess(f'{editor_program}', f'"{__file__}"')
        elif event == '-FILTER-':
            new_list = [i for i in file_list if values['-FILTER-'].lower() in i.lower()]
            window['-DEMO LIST-'].update(new_list)
            window['-FILTER NUMBER-'].update(f'{len(new_list)} files')
            window['-FIND NUMBER-'].update('')
            window['-FIND-'].update('')
            window['-FIND RE-'].update('')
        elif event == '-FIND-' or event == '-FIRST MATCH ONLY-' or event == '-VERBOSE-' or (event == '-FIND RE-'):
            is_ignore_case = values['-IGNORE CASE-']
            old_ignore_case = False
            current_typed_value = str(values['-FIND-'])
            if len(values['-FIND-']) == 1:
                window[ML_KEY].update('')
                window['-VERBOSE-'].update(False)
                values['-VERBOSE-'] = False
            if values['-VERBOSE-']:
                window[ML_KEY].update('')
            if values['-FIND-']:
                if find_in_file.old_file_list is None or old_typed_value is None or old_ignore_case is not is_ignore_case:
                    old_typed_value = current_typed_value
                    file_list = find_in_file(values['-FIND-'], get_file_list_dict(), verbose=values['-VERBOSE-'], window=window, ignore_case=is_ignore_case, show_first_match=values['-FIRST MATCH ONLY-'])
                elif current_typed_value.startswith(old_typed_value) and old_ignore_case is is_ignore_case:
                    old_typed_value = current_typed_value
                    file_list = find_in_file(values['-FIND-'], find_in_file.old_file_list, verbose=values['-VERBOSE-'], window=window, ignore_case=is_ignore_case, show_first_match=values['-FIRST MATCH ONLY-'])
                else:
                    old_typed_value = current_typed_value
                    file_list = find_in_file(values['-FIND-'], get_file_list_dict(), verbose=values['-VERBOSE-'], window=window, ignore_case=is_ignore_case, show_first_match=values['-FIRST MATCH ONLY-'])
                window['-DEMO LIST-'].update(sorted(file_list))
                window['-FIND NUMBER-'].update(f'{len(file_list)} files')
                window['-FILTER NUMBER-'].update('')
                window['-FIND RE-'].update('')
                window['-FILTER-'].update('')
            elif values['-FIND RE-']:
                window['-ML-'].update('')
                file_list = find_in_file(values['-FIND RE-'], get_file_list_dict(), regex=True, verbose=values['-VERBOSE-'], window=window)
                window['-DEMO LIST-'].update(sorted(file_list))
                window['-FIND NUMBER-'].update(f'{len(file_list)} files')
                window['-FILTER NUMBER-'].update('')
                window['-FIND-'].update('')
                window['-FILTER-'].update('')
        elif event == 'Find RE':
            window['-ML-'].update('')
            file_list = find_in_file(values['-FIND RE-'], get_file_list_dict(), regex=True, verbose=values['-VERBOSE-'], window=window)
            window['-DEMO LIST-'].update(sorted(file_list))
            window['-FIND NUMBER-'].update(f'{len(file_list)} files')
            window['-FILTER NUMBER-'].update('')
            window['-FIND-'].update('')
            window['-FILTER-'].update('')
            sg.cprint('Regular expression find completed')
        elif event == 'Settings':
            if settings_window() is True:
                window.close()
                window = make_window()
                file_list_dict = get_file_list_dict()
                file_list = get_file_list()
                window['-FILTER NUMBER-'].update(f'{len(file_list)} files')
        elif event == 'Clear':
            file_list = get_file_list()
            window['-FILTER-'].update('')
            window['-FILTER NUMBER-'].update(f'{len(file_list)} files')
            window['-FIND-'].update('')
            window['-DEMO LIST-'].update(file_list)
            window['-FIND NUMBER-'].update('')
            window['-FIND RE-'].update('')
            window['-ML-'].update('')
        elif event == '-FOLDERNAME-':
            sg.user_settings_set_entry('-demos folder-', values['-FOLDERNAME-'])
            file_list_dict = get_file_list_dict()
            file_list = get_file_list()
            window['-DEMO LIST-'].update(values=file_list)
            window['-FILTER NUMBER-'].update(f'{len(file_list)} files')
            window['-ML-'].update('')
            window['-FIND NUMBER-'].update('')
            window['-FIND-'].update('')
            window['-FIND RE-'].update('')
            window['-FILTER-'].update('')
        elif event == 'Open Folder':
            explorer_program = get_explorer()
            if explorer_program:
                sg.cprint(f'Opening Folder using {explorer_program}...', c='white on green', end='')
                sg.cprint('')
                for file in values['-DEMO LIST-']:
                    file_selected = str(file_list_dict[file])
                    file_path = os.path.dirname(file_selected)
                    if running_windows():
                        file_path = file_path.replace('/', '\\')
                    sg.cprint(file_path, text_color='white', background_color='purple')
                    execute_command_subprocess(explorer_program, file_path)
    window.close()
try:
    execute_py_file = sg.execute_py_file
except:

    def execute_py_file(pyfile, parms=None, cwd=None):
        if False:
            return 10
        if parms is not None:
            execute_command_subprocess('python' if running_windows() else 'python3', pyfile, parms, wait=False, cwd=cwd)
        else:
            execute_command_subprocess('python' if running_windows() else 'python3', pyfile, wait=False, cwd=cwd)
try:
    execute_command_subprocess = sg.execute_command_subprocess
except:

    def execute_command_subprocess(command, *args, wait=False, cwd=None):
        if False:
            i = 10
            return i + 15
        if running_linux():
            arg_string = ''
            for arg in args:
                arg_string += ' ' + str(arg)
            sp = subprocess.Popen(str(command) + arg_string, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=cwd)
        else:
            expanded_args = ' '.join(args)
            sp = subprocess.Popen([command, *args], shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=cwd)
        if wait:
            (out, err) = sp.communicate()
            if out:
                print(out.decode('utf-8'))
            if err:
                print(err.decode('utf-8'))
if __name__ == '__main__':
    icon = b'iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAYAAABzenr0AAAACXBIWXMAAAsSAAALEgHS3X78AAAK/2lUWHRYTUw6Y29tLmFkb2JlLnhtcAAAAAAAPD94cGFja2V0IGJlZ2luPSLvu78iIGlkPSJXNU0wTXBDZWhpSHpyZVN6TlRjemtjOWQiPz4NCjx4OnhtcG1ldGEgeG1sbnM6eD0iYWRvYmU6bnM6bWV0YS8iIHg6eG1wdGs9IkFkb2JlIFhNUCBDb3JlIDUuMy1jMDA3IDEuMTM2ODgxLCAyMDEwLzA2LzEwLTE4OjExOjM1ICAgICAgICAiPg0KICA8cmRmOlJERiB4bWxuczpyZGY9Imh0dHA6Ly93d3cudzMub3JnLzE5OTkvMDIvMjItcmRmLXN5bnRheC1ucyMiPg0KICAgIDxyZGY6RGVzY3JpcHRpb24gcmRmOmFib3V0PSIiIHhtbG5zOnhtcE1NPSJodHRwOi8vbnMuYWRvYmUuY29tL3hhcC8xLjAvbW0vIiB4bWxuczpzdEV2dD0iaHR0cDovL25zLmFkb2JlLmNvbS94YXAvMS4wL3NUeXBlL1Jlc291cmNlRXZlbnQjIiB4bWxuczpkYz0iaHR0cDovL3B1cmwub3JnL2RjL2VsZW1lbnRzLzEuMS8iIHhtbG5zOnhtcD0iaHR0cDovL25zLmFkb2JlLmNvbS94YXAvMS4wLyIgeG1wTU06RG9jdW1lbnRJRD0iRUM4REZFNUEyMEM0QjcwMzFBMjNBRDA4NENCNzZCODAiIHhtcE1NOk9yaWdpbmFsRG9jdW1lbnRJRD0iRUM4REZFNUEyMEM0QjcwMzFBMjNBRDA4NENCNzZCODAiIHhtcE1NOkluc3RhbmNlSUQ9InhtcC5paWQ6OUVBNkIyNzA3NjYyRTkxMThDQzNFODdFN0ZEQTgwNTEiIGRjOmZvcm1hdD0iaW1hZ2UvanBlZyIgeG1wOlJhdGluZz0iNSIgeG1wOk1ldGFkYXRhRGF0ZT0iMjAxOS0wNC0xOVQxMjoxMTozOSswNDozMCI+DQogICAgICA8eG1wTU06SGlzdG9yeT4NCiAgICAgICAgPHJkZjpTZXE+DQogICAgICAgICAgPHJkZjpsaSBzdEV2dDphY3Rpb249InNhdmVkIiBzdEV2dDppbnN0YW5jZUlEPSJ4bXAuaWlkOkYwNjRBQzcwNzY2MkU5MTFBNkU1QzYwODhFRTUxMzM5IiBzdEV2dDp3aGVuPSIyMDE5LTA0LTE5VDEyOjExOjM5KzA0OjMwIiBzdEV2dDpzb2Z0d2FyZUFnZW50PSJBZG9iZSBQaG90b3Nob3AgQ2FtZXJhIFJhdyA3LjAiIHN0RXZ0OmNoYW5nZWQ9Ii9tZXRhZGF0YSIgLz4NCiAgICAgICAgICA8cmRmOmxpIHN0RXZ0OmFjdGlvbj0ic2F2ZWQiIHN0RXZ0Omluc3RhbmNlSUQ9InhtcC5paWQ6OUVBNkIyNzA3NjYyRTkxMThDQzNFODdFN0ZEQTgwNTEiIHN0RXZ0OndoZW49IjIwMTktMDQtMTlUMTI6MTE6MzkrMDQ6MzAiIHN0RXZ0OnNvZnR3YXJlQWdlbnQ9IkFkb2JlIFBob3Rvc2hvcCBDYW1lcmEgUmF3IDcuMCAoV2luZG93cykiIHN0RXZ0OmNoYW5nZWQ9Ii9tZXRhZGF0YSIgLz4NCiAgICAgICAgPC9yZGY6U2VxPg0KICAgICAgPC94bXBNTTpIaXN0b3J5Pg0KICAgICAgPGRjOnRpdGxlPg0KICAgICAgICA8cmRmOkFsdD4NCiAgICAgICAgICA8cmRmOmxpIHhtbDpsYW5nPSJ4LWRlZmF1bHQiPkJ1bGIgSWNvbiBEZXNpZ248L3JkZjpsaT4NCiAgICAgICAgPC9yZGY6QWx0Pg0KICAgICAgPC9kYzp0aXRsZT4NCiAgICAgIDxkYzpjcmVhdG9yPg0KICAgICAgICA8cmRmOlNlcT4NCiAgICAgICAgICA8cmRmOmxpPklZSUtPTjwvcmRmOmxpPg0KICAgICAgICA8L3JkZjpTZXE+DQogICAgICA8L2RjOmNyZWF0b3I+DQogICAgICA8ZGM6ZGVzY3JpcHRpb24+DQogICAgICAgIDxyZGY6QWx0Pg0KICAgICAgICAgIDxyZGY6bGkgeG1sOmxhbmc9IngtZGVmYXVsdCI+QnVsYiBJY29uIERlc2lnbg0KPC9yZGY6bGk+DQogICAgICAgIDwvcmRmOkFsdD4NCiAgICAgIDwvZGM6ZGVzY3JpcHRpb24+DQogICAgICA8ZGM6c3ViamVjdD4NCiAgICAgICAgPHJkZjpCYWc+DQogICAgICAgICAgPHJkZjpsaT5idWxiPC9yZGY6bGk+DQogICAgICAgICAgPHJkZjpsaT5lbmVyZ3k8L3JkZjpsaT4NCiAgICAgICAgICA8cmRmOmxpPmlkZWE8L3JkZjpsaT4NCiAgICAgICAgICA8cmRmOmxpPmxpZ2h0PC9yZGY6bGk+DQogICAgICAgICAgPHJkZjpsaT5saWdodGJ1bGI8L3JkZjpsaT4NCiAgICAgICAgICA8cmRmOmxpPmJ1bGIgaWNvbjwvcmRmOmxpPg0KICAgICAgICAgIDxyZGY6bGk+ZW5lcmd5IGljb248L3JkZjpsaT4NCiAgICAgICAgICA8cmRmOmxpPmlkZWEgaWNvbjwvcmRmOmxpPg0KICAgICAgICAgIDxyZGY6bGk+bGlnaHQgaWNvbjwvcmRmOmxpPg0KICAgICAgICAgIDxyZGY6bGk+bGlnaHRidWxiIGljb248L3JkZjpsaT4NCiAgICAgICAgICA8cmRmOmxpPmljb248L3JkZjpsaT4NCiAgICAgICAgICA8cmRmOmxpPmlsbHVzdHJhdGlvbjwvcmRmOmxpPg0KICAgICAgICAgIDxyZGY6bGk+ZGVzaWduPC9yZGY6bGk+DQogICAgICAgICAgPHJkZjpsaT5zaWduPC9yZGY6bGk+DQogICAgICAgICAgPHJkZjpsaT5zeW1ib2w8L3JkZjpsaT4NCiAgICAgICAgICA8cmRmOmxpPmdyYXBoaWM8L3JkZjpsaT4NCiAgICAgICAgICA8cmRmOmxpPmxpbmU8L3JkZjpsaT4NCiAgICAgICAgICA8cmRmOmxpPmxpbmVhcjwvcmRmOmxpPg0KICAgICAgICAgIDxyZGY6bGk+b3V0bGluZTwvcmRmOmxpPg0KICAgICAgICAgIDxyZGY6bGk+ZmxhdDwvcmRmOmxpPg0KICAgICAgICAgIDxyZGY6bGk+Z2x5cGg8L3JkZjpsaT4NCiAgICAgICAgICA8cmRmOmxpPmdyYWRpZW50PC9yZGY6bGk+DQogICAgICAgICAgPHJkZjpsaT5jaXJjbGU8L3JkZjpsaT4NCiAgICAgICAgICA8cmRmOmxpPnNoYWRvdzwvcmRmOmxpPg0KICAgICAgICAgIDxyZGY6bGk+bG93IHBvbHk8L3JkZjpsaT4NCiAgICAgICAgICA8cmRmOmxpPnBvbHlnb25hbDwvcmRmOmxpPg0KICAgICAgICAgIDxyZGY6bGk+c3F1YXJlPC9yZGY6bGk+DQogICAgICAgIDwvcmRmOkJhZz4NCiAgICAgIDwvZGM6c3ViamVjdD4NCiAgICA8L3JkZjpEZXNjcmlwdGlvbj4NCiAgPC9yZGY6UkRGPg0KPC94OnhtcG1ldGE+DQo8P3hwYWNrZXQgZW5kPSJyIj8+ncdlNwAABG5JREFUWMPtV2tMk1cYfr6vINBRKpVLpVIuGZfOjiARochYO/EyMWHDjOIYYSyQ6MaignEkEgJLyNxMRthmJBkalKmgUacsjMiCBRFWqKsBFJEwtkILLYxBuQi0/c5+sE5mJmvBxR/zJO+fc877vE/ey5NzKEIInuWi8YzXcwIrInCtuirlWnVVykowHGy9qFYqo+bn5pyj4uIarXtBope6H7+nbGp6dZWT0+yGqCjlUyUQEBTUW15SkiOOiLj9/eXLu1sVN6Q6jUYIAD5CoUYilSleT0q6dLO+fmvmwYOfP/UScN3df+cLBIPvbN92fWpijJedtk5XWQT6TM5PkR9Izrw72ZpRkRrr/yvfhz/MXe02aSsuZYsOMAxDH87MPMlfJxjcn7OnitWReg7GjrDH75ktQGkVMDz/csdnFReSWZzgnmVnwGwyOSbLpI1davWGY/n5xaFhYR2HPko/zWrb0vBPwQHAgQXkpgKhvM6wY+/HNXU1X0xOlkkbzSaT4xMZEEKeaDPT02xVy62YrKQ3rxDGQltubmq31NDEFsuUUKTthPjezNQkZ6kYS/aAC5s9U1lWti+36OMCMvxtEsZVG22tbU4qhW8u3hU5G69vX3YTMgxDD/T3B4SIxZ1EVyW3Z75D/IABPcBoq+XLJjA0MOArDAj4GQAwcSfCXpERegO6PnXEsglMTU1xXN3cjAtdaXSz7s/MAl19gHZKqNGOAF0634GZOQcz3GNaF/u7soHpyUd+dhPw8PQ06HVDPgAA57W6v2aXAgrKCBrazKyGzrVDBSfmHSn/vWXU6si2xf6GMWCN9yM/u5VwjZeXYdSg9559+JDt5LWzlvw5fi5OQFoChS+qtAK88GLv/rzs4+zASBVpkd2w+s7OASPjgEfQztoVCdH5k+WZo3q994e5WV8zivXdMI3xFsYX2H2YAC4C7ZXbGl/SvqsWhrodVr+vLgAeXryxt4vviuDkZVi2FMsz3julutWyuV31IJgKr0gHxbJYyyDfRkH+ik6AcWX04uDt9wDVfdoiP1SRvlRwmwjQNM2UVlamfZKXd7T3t4B+SvxltvWMRS8YmDkn616vBvj0NFB66ng2i5/w3b+OylIqtdi0Go0wMUbSOjQ4KGB6CossNTSpPrBgzKhCaqmhibaCJokiigw2FxbZimszAUIIutTq8D3xW36wmE0OFmVC7WICpqs0SQmnSOf5hFpCGMpWTAd7hGV9ePgdiVSmyNu7r8zPx3egU7XQwCOl51J/URJItr5xVZxYcgCgbH9q25MBQgiM4+NcmUjUrair23FJFtt81hnkrDPIZhZIf37eUXvx7H4TcrjcCZ6nx0hsfHx9nEzWsJEGImiAC8B1leP8f/Ym/JvEctwm5a/JFMyQzsc0T4EBwIuK/pGbkVVuN5i9KSOE4C2ZVMFYLPRI4ZHiHjbIfTbILhbISOGRYnuxqOV8zaL9/TR+gYF98w96Qs3DQ3wA0HO4xmalctOq4JAee7Co53/D/z2BPwAlMRlLdQS6SQAAAABJRU5ErkJggg=='
    main()