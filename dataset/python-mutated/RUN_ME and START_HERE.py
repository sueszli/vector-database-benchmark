import time
import subprocess, re, datetime, time, os, platform, json, PySimpleGUI as sg
from subprocess import Popen
from make_real_readme import main
import os
cd = CD = os.path.dirname(os.path.abspath(__file__))
dir_name = os.path.join(cd, 'output')
if not os.path.exists(dir_name):
    os.mkdir(dir_name)
else:
    print(f'Такая папка уже есть: "{dir_name}"')
sg.theme('Dark2')
cd = os.path.dirname(os.path.abspath(__file__))

def readfile(filename):
    if False:
        i = 10
        return i + 15
    with open(filename, 'r', encoding='utf-8') as ff:
        return ff.read()

def writefile(fpath, content):
    if False:
        print('Hello World!')
    with open(fpath, 'w', encoding='utf-8') as ff:
        ff.write(content)

def writejson(a_path: str, a_dict: dict) -> None:
    if False:
        while True:
            i = 10
    with open(a_path, 'w', encoding='utf-8') as output_file:
        json.dump(a_dict, output_file, ensure_ascii=False, indent=2)

def readjson(a_path: str) -> dict:
    if False:
        while True:
            i = 10
    with open(a_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def openfile(a_path):
    if False:
        for i in range(10):
            print('nop')
    if not os.path.exists(a_path):
        return sg.Popup(f"Error! This file doesn't exists: {a_path}")
    if 'Windows' in platform.system():
        os.startfile(a_path)
    elif 'Linux' in platform.system():
        Popen(f'exo-open "{a_path}"', shell=True)

def opendir(a_path):
    if False:
        for i in range(10):
            print('nop')
    if not os.path.exists(a_path):
        return sg.Popup(f"Error! This directory doesn't exists: {a_path}")
    try:
        if 'Windows' in platform.system():
            os.startfile(a_path)
        elif 'Linux' in platform.system():
            Popen(f'exo-open --launch FileManager --working-directory "{a_path}"', shell=True)
    except Exception as e:
        sg.Popen(f"Error, can't open a file: '{e}'")

def load_configs():
    if False:
        i = 10
        return i + 15
    return readjson(os.path.join(cd, 'app_configs.json'))

def save_configs(a_config: dict):
    if False:
        print('Hello World!')
    writejson(os.path.join(cd, 'app_configs.json'), a_config)
APP_CONFIGS = load_configs()
README_OFILENAME = APP_CONFIGS['README_OFILE']
CALL_REFERENCE_OFILENAME = APP_CONFIGS['CALL_REF_OFILE']
insert_md_section_for__class_methods = False
remove_repeated_sections_classmethods = False
import time

def timeit(f):
    if False:
        return 10

    def wrapper(*args, **kwargs):
        if False:
            i = 10
            return i + 15
        start = time.time()
        res = f(*args, **kwargs)
        end = time.time()
        return res
    return wrapper

class BESTLOG(object):

    def __init__(self, filename):
        if False:
            i = 10
            return i + 15
        self.filename = filename
        self.json_name = filename + '.json'
        self.error_list = []
        self.warning_list = []
        self.info_list = []
        self.debug_list = []
        self.tick_amount = 1
        self.names = self.messages_names = 'error warning info debug'.split(' ')

    def tick(self):
        if False:
            print('Hello World!')
        self.tick_amount += 1
        return self.tick_amount

    def error(self, m, metadata={}):
        if False:
            return 10
        self.error_list.append([self.tick(), m, metadata])

    def warning(self, m, metadata={}):
        if False:
            for i in range(10):
                print('nop')
        self.warning_list.append([self.tick(), m, metadata])

    def info(self, m, metadata={}):
        if False:
            i = 10
            return i + 15
        self.info_list.append([self.tick(), m, metadata])

    def debug(self, m, metadata={}):
        if False:
            while True:
                i = 10
        self.debug_list.append([self.tick(), m, metadata])

    def tolist(self):
        if False:
            for i in range(10):
                print('nop')
        return zip([self.error_list, self.warning_list, self.info_list, self.debug_list], self.names)

    def todict(self):
        if False:
            for i in range(10):
                print('nop')
        return {'error': self.error_list, 'warning': self.warning_list, 'info': self.info_list, 'debug': self.debug_list}

    @timeit
    def save(self):
        if False:
            while True:
                i = 10
        "\n\t\t{\n\t\t\t'message_type' : message_type,\n\t\t\t'message_text' : m_text,\n\t\t\t'message_time' : m_time,\n\t\t\t'message_metadata' : m_metadata\n\t\t}\n\t\t"
        all_messages_list = []
        for (messages, message_type) in self.tolist():
            results_ = [{'message_type': message_type, 'message_text': m_text, 'message_time': m_time, 'message_metadata': m_metadata} for (m_time, m_text, m_metadata) in messages]
            all_messages_list.extend(results_)
        all_messages_list = sorted(all_messages_list, key=lambda x: x['message_time'])
        writejson(self.json_name, all_messages_list)

    @timeit
    def load(self, **kw):
        if False:
            while True:
                i = 10
        '\n\t\t\treturn dict with messages\n\t\t\t\n\t\t\tkw = {\n\t\t\t\tuse_psg_color : bool\n\t\t\t\tshow_time : bool\n\t\t\t}\n\t\t'
        all_messages_list = readjson(self.json_name)

        def format_message(message):
            if False:
                i = 10
                return i + 15
            if kw['show_time']:
                return str(message['message_time']) + ':' + message['message_text']
            else:
                return message['message_text']
        error_list = [i for i in all_messages_list if i['message_type'] == 'error']
        warning_list = [i for i in all_messages_list if i['message_type'] == 'warning']
        info_list = [i for i in all_messages_list if i['message_type'] == 'info']
        debug_list = [i for i in all_messages_list if i['message_type'] == 'debug']
        colors = {'warning': 'blue', 'info': 'black'}
        warning_info_ = []
        for message in sorted(warning_list + info_list, key=lambda x: x['message_time']):
            if kw['use_psg_color']:
                warning_info_.append([format_message(message), colors.get(message['message_type'])])
            else:
                warning_info_.append(format_message(message))
        error_list = [format_message(i) for i in error_list]
        warning_list = [format_message(i) for i in warning_list]
        info_list = [format_message(i) for i in info_list]
        debug_list = [format_message(i) for i in debug_list]
        return (error_list, warning_list, info_list, debug_list, warning_info_)

    @timeit
    def load_to_listbox(self):
        if False:
            return 10
        '\n\t\tread .json\n\t\t'
        return sorted(readjson(self.json_name), key=lambda x: x['message_time'])

@timeit
def compile_call_ref(output_filename='LoG_call_ref', **kw):
    if False:
        while True:
            i = 10
    ' Compile a "5_call_reference.md" file'
    log_obj = BESTLOG(os.path.join(cd, output_filename))
    main(logger=log_obj, main_md_file='markdown input files/5_call_reference.md', insert_md_section_for__class_methods=insert_md_section_for__class_methods, remove_repeated_sections_classmethods=remove_repeated_sections_classmethods, files_to_include=[], output_name=CALL_REFERENCE_OFILENAME, delete_html_comments=True)
    log_obj.save()
    return (log_obj.load(**kw), log_obj.load_to_listbox())

@timeit
def compile_readme(output_filename='LoG', **kw):
    if False:
        for i in range(10):
            print('nop')
    ' Compile a "2_readme.md" file'
    log_obj = BESTLOG(os.path.join(cd, output_filename))
    main(logger=log_obj, insert_md_section_for__class_methods=insert_md_section_for__class_methods, remove_repeated_sections_classmethods=remove_repeated_sections_classmethods, files_to_include=[0, 1, 2, 3], output_name=README_OFILENAME, delete_html_comments=True)
    log_obj.save()
    return (log_obj.load(**kw), log_obj.load_to_listbox())

def compile_all_stuff(**kw):
    if False:
        i = 10
        return i + 15
    '\n\t\tCompile a "2_ and 5_" .md filess\n\t\treturn output from them\n\t'
    return (compile_readme(**kw), compile_call_ref(**kw))

def md2psg(target_text):
    if False:
        while True:
            i = 10
    "\n\t\tib<space>color\n\t\ti italic\n\t\tb bold\n\t\tcolor = can be word   can be color\n\t\t\t\tred             #ff00111\n\t\t\t\tgreen\n\t\t\t\tblue\n\t\ti?b?\\s?\\w+?\n\n\n\t\tusage\n\t\t  *i*a**            italic\n\t\t  *b*a**            bold\n\t\t  *ib*a**           italic bold\n\t\t  *ib red*a**       italic bold red\n\t\t  *b green*a**      bold green\n\t\t\n\t\t'This was *I*special** message from *B*him**. And from *Igreen*this** to *Ired*this**'\n\t"
    font_norm = 'Mono 12 '
    font_bold = 'Mono 12 italic'
    font_italic = 'Mono 12 bold'
    list_of_Ts = []
    parts = [i for i in re.compile('(\\*I?B?[a-z]*?\\*[\\d\\D]*?\\*\\*)', flags=re.M | re.DOTALL).split(target_text) if i is not None]
    for (index, text) in enumerate(parts):
        if index % 2 == 0:
            T_text = text
            T = sg.T(T_text, size=(len(T_text), 1), pad=(0, 0), font=font_norm)
        else:
            T_parameters = {'font': font_norm}
            my_format = text[1:].split('*')[0]
            if 'I' in my_format:
                T_parameters['font'] = font_italic
            if 'B' in my_format:
                T_parameters['font'] = font_bold
            color_left = my_format.replace('I', '').replace('B', '')
            if color_left:
                T_parameters['text_color'] = color_left
            T_text = '*'.join(text.split('*')[2:-2])
            T = sg.T(T_text, size=(len(T_text), 1), pad=(0, 0), **T_parameters)
        list_of_Ts.append(T)
    return list_of_Ts

def mini_GUI():
    if False:
        for i in range(10):
            print('nop')
    my_font = ('Helvetica', 12)
    my_font2 = ('Helvetica', 12, 'bold')
    my_font3 = ('Helvetica', 15, 'bold')
    my_font4 = ('Mono', 18, 'bold')

    def make_tab(word):
        if False:
            i = 10
            return i + 15

        def tabs(*layouts):
            if False:
                while True:
                    i = 10
            return sg.TabGroup([[sg.Tab(title, lay, key=f'-tab-{word_}-{index}-') for (index, (title, word_, lay)) in enumerate(layouts)]])
        return [[sg.Column(layout=[[sg.T('debug', font=my_font, text_color='grey')], [sg.ML(size=(50 - 15, 15), key=f'-{word}-debug-')], [sg.T('error', font=my_font, text_color='red')], [sg.ML(size=(50 - 15, 15), key=f'-{word}-error-')]], pad=(0, 0)), sg.T('    '), sg.Column(layout=[[sg.T('warning', font=my_font2)], [sg.ML(size=(70 - 12, 15), key=f'-{word}-warning-')], [sg.T('info', font=my_font2)], [sg.ML(size=(70 - 12, 15), key=f'-{word}-info-')]], pad=(0, 0)), tabs(('Text', word, [[sg.T('warning info', font=my_font3)], [sg.ML(size=(110, 30), key=f'-{word}-warning_info-')]]), ('Listbox', word, [[sg.T('warning info listbox', font=my_font3)], [sg.Listbox([], size=(110, 30 - 1), key=f'-{word}-listbox-', enable_events=True, background_color='#ffccaa')]]))]]
    settings_layout = [[sg.CB('Toggle progressbar', False, enable_events=True, key='toggle_progressbar')], [sg.Frame('Text editor', [[sg.Combo(['pycharm', 'subl'], default_value='subl', enable_events=True, key='_text_editor_combo_')]]), sg.Frame('Pycharm path:', [[sg.I('', size=(40, 1), enable_events=True, key='_PyCharm_path_')]])], [sg.Frame('⅀∉ Filter "empty tables"', [[sg.T('This is for filtering stirng, like:')], [sg.T('Warning =======    We got empty md_table for "EasyPrintClose"', font='Mono 8')], [sg.CB('enable', True, key='checkbox_enable_empty_tables_filter', enable_events=True)], [sg.ML('PrintClose\nEasyPrintClose\nmain\ntheme\nRead', size=(30, 10), enable_events=True, key='_filter_empty_tables_ml_')]]), sg.Frame('⅀∉ Filter "tkinter class methods"', [[sg.T('This is for filtering stirng, like:')], [sg.T("Please, fix ':return:' in 'SetFocus'                  IF you want to see 'return' row in 'signature table' ", font='Mono 8')], [sg.CB('enable', True, enable_events=True, key='checkbox_enable_filter_tkinter_class_methods')], [sg.ML('SetFocus\nSetTooltip\nUpdate\n__init__\nbind\nexpand\nset_cursor\nset_size', size=(30, 10), enable_events=True, key='_filter_tkinter_class_methods_')]], visible=not True)]]
    layout = [[sg.TabGroup([[sg.Tab('readme logs', make_tab('README')), sg.Tab('Call reference logs', make_tab('CALL_REF')), sg.Tab('General settings', settings_layout)]])]]
    from time import sleep
    from math import pi, sin
    from itertools import count

    def next_star():
        if False:
            i = 10
            return i + 15
        middle = 100 / 2
        for i in (int(sin(i * pi / middle) * middle + middle) for i in count()):
            yield i
    psg_module_path = str(sg).split("' from '")[1][:-2]
    star_bar = sg.Col([[sg.ProgressBar(max_value=100, orientation='h', key='_star_bar1_', size=(50, 5), bar_color=('blue', 'yellow'))], [sg.ProgressBar(max_value=100, orientation='h', key='_star_bar2_', size=(50, 5), bar_color=('yellow', 'blue'))]])

    def empty_line(fontsize=12):
        if False:
            i = 10
            return i + 15
        return [sg.T('', font='Mono ' + str(fontsize))]
    window = sg.Window('We are live! Again! --- ' + 'Completed making            {}, {}'.format(os.path.basename(README_OFILENAME), os.path.basename(CALL_REFERENCE_OFILENAME)), [[sg.T(size=(30, 1), key='-compile-time-'), star_bar], empty_line(), [*md2psg(f'The *Bmagenta*PySimpleGUI** module being processed is *Imagenta*"{psg_module_path}"**'), sg.B('< open (__init__.py)', key='open_init_file'), sg.B('< open (psg.py)', key='open_psg_file')], empty_line(), [sg.B('Run again (F1)', key='-run-'), sg.Col([[sg.CB('show time in logs (F2)', False, enable_events=True, key='show_time')], [sg.CB('Logs with Color (F3)', True, enable_events=True, key='use_psg_color')]]), sg.Col([empty_line(5), [sg.B('open "db folder"', key='-open_db_folder-')]]), sg.Frame('', [[sg.Col([[*md2psg('markdown outputFileName *I*FOR** *B*readme  **: '), sg.I(README_OFILENAME, key='README_OFILE', size=(25, 1)), sg.B('open in explorer', key='open in explorer_readme'), sg.B('open in text editor', key='open file - readme')], [*md2psg('markdown outputFileName *I*FOR** *B*call ref**: '), sg.I(CALL_REFERENCE_OFILENAME, key='CALL_REF_OFILE', size=(25, 1)), sg.B('open in explorer', key='open in explorer_calref'), sg.B('open in text editor', key='open file - calref')]])]], relief=sg.RELIEF_SUNKEN, border_width=4)], *layout], resizable=True, finalize=True, location=(0, 0), return_keyboard_events=True)

    def update_time_in_GUI():
        if False:
            return 10
        window['-compile-time-'](datetime.datetime.today().strftime('%Y-%m-%d %H:%M:%S.%f'))

    def update_compilation_in_psg(values):
        if False:
            print('Hello World!')
        (result_readme__for_txt_n_listbox, result_call_ref__for_txt_n_listbox) = compile_all_stuff(use_psg_color=values['use_psg_color'], show_time=values['show_time'])
        (result_readme_txt, result_readme_listbox_items) = result_readme__for_txt_n_listbox
        (result_call_ref_txt, result_call_ref_listbox_items) = result_call_ref__for_txt_n_listbox
        badNames = [i.strip() for i in values['_filter_tkinter_class_methods_'].split('\n') if i.strip()]
        badNames = '|'.join(badNames)
        regex_str1 = f'fix .:return:. in .({badNames}).'
        badNames = [i for i in values['_filter_empty_tables_ml_'].split('\n') if i.strip()]
        badNames = '|'.join(badNames)
        regex_str2 = f'empty md_table for .({badNames}).'

        def is_valid_regex_LogMessage(msg: str):
            if False:
                for i in range(10):
                    print('nop')
            nonlocal regex_str1, regex_str2
            error1_found = False
            if values['checkbox_enable_filter_tkinter_class_methods'] and ':return:' in msg:
                error1_found = bool(re.search(regex_str1, msg, flags=re.M | re.DOTALL))
            error2_found = False
            if values['checkbox_enable_empty_tables_filter'] and 'empty md_table for' in msg:
                error2_found = bool(re.search(regex_str2, msg, flags=re.M | re.DOTALL))
            return not error1_found and (not error2_found)

        def filter_log_messages(messages):
            if False:
                i = 10
                return i + 15
            if type(messages) is str:
                return '\n'.join([msg for msg in messages.split('\n') if is_valid_regex_LogMessage(msg)])
            raise TypeError

        class ParsingError(object):

            def __init__(self, log_obj):
                if False:
                    i = 10
                    return i + 15
                self.log_obj = log_obj
                self.text = log_obj['message_text']

            def __str__(self):
                if False:
                    for i in range(10):
                        print('nop')
                return self.__repr__()

            def __repr__(self):
                if False:
                    return 10
                'qwe'
                text = self.log_obj['message_text']
                metadata = self.log_obj['message_metadata']
                lineno = ''
                if 'lineno' in metadata.keys():
                    lineno = '(line:' + str(metadata['lineno']) + ') '
                return f'{lineno} {text}'
        items1 = [i for i in result_readme_listbox_items if is_valid_regex_LogMessage(i['message_text'])]
        items2 = [i for i in result_call_ref_listbox_items if is_valid_regex_LogMessage(i['message_text'])]
        window['-README-listbox-']([ParsingError(i) for i in items1])
        window['-CALL_REF-listbox-']([ParsingError(i) for i in items2])

        def set_it(prefix='CALL_REF', messages_obj=result_call_ref_txt):
            if False:
                i = 10
                return i + 15
            (t_error, t_warning, t_info, t_debug) = ['\n'.join(i) for i in messages_obj[:4]]
            t_error = filter_log_messages(t_error)
            t_warning = filter_log_messages(t_warning)
            t_info = filter_log_messages(t_info)
            t_debug = filter_log_messages(t_debug)
            window[f'-{prefix}-error-'](t_error)
            window[f'-{prefix}-warning-'](t_warning)
            window[f'-{prefix}-info-'](t_info)
            window[f'-{prefix}-debug-'](t_debug)
            window[f'-{prefix}-warning_info-'].update('')
            t_warning_info_obj = messages_obj[-1]
            if values['use_psg_color']:
                for (text, color) in t_warning_info_obj:
                    if not is_valid_regex_LogMessage(text):
                        continue
                    window[f'-{prefix}-warning_info-'].print(text, text_color=color)
            else:
                window[f'-{prefix}-warning_info-'](t_warning_info_obj)
        set_it('README', result_readme_txt)
        set_it('CALL_REF', result_call_ref_txt)
        update_time_in_GUI()
    values = window.read(timeout=0)[1]
    update_compilation_in_psg(values)
    p_values = values
    window['_PyCharm_path_'](APP_CONFIGS['_PyCharm_path_'])
    window['_text_editor_combo_'].update(set_to_index=APP_CONFIGS['_text_editor_combo_'])
    window['toggle_progressbar'](APP_CONFIGS['toggle_progressbar'])
    window['checkbox_enable_empty_tables_filter'](APP_CONFIGS['checkbox_enable_empty_tables_filter'])
    window['_filter_empty_tables_ml_'](APP_CONFIGS['_filter_empty_tables_ml_'])
    window['checkbox_enable_filter_tkinter_class_methods'](APP_CONFIGS['checkbox_enable_filter_tkinter_class_methods'])
    window['_filter_tkinter_class_methods_'](APP_CONFIGS['_filter_tkinter_class_methods_'])
    window['show_time'](APP_CONFIGS['show_time'])
    window['use_psg_color'](APP_CONFIGS['use_psg_color'])
    window['README_OFILE'](APP_CONFIGS['README_OFILE'])
    window['CALL_REF_OFILE'](APP_CONFIGS['CALL_REF_OFILE'])
    next_val_gen = next_star()
    my_timeout = None
    while True:
        (event, values) = window(timeout=my_timeout)
        if event in ('Exit', None):
            APP_CONFIGS['_text_editor_combo_'] = 1 if window['_text_editor_combo_'].get() == 'subl' else 0
            APP_CONFIGS['toggle_progressbar'] = p_values['toggle_progressbar']
            APP_CONFIGS['checkbox_enable_empty_tables_filter'] = p_values['checkbox_enable_empty_tables_filter']
            APP_CONFIGS['_filter_empty_tables_ml_'] = p_values['_filter_empty_tables_ml_']
            APP_CONFIGS['checkbox_enable_filter_tkinter_class_methods'] = p_values['checkbox_enable_filter_tkinter_class_methods']
            APP_CONFIGS['_filter_tkinter_class_methods_'] = p_values['_filter_tkinter_class_methods_']
            APP_CONFIGS['show_time'] = p_values['show_time']
            APP_CONFIGS['use_psg_color'] = p_values['use_psg_color']
            APP_CONFIGS['README_OFILE'] = p_values['README_OFILE']
            APP_CONFIGS['CALL_REF_OFILE'] = p_values['CALL_REF_OFILE']
            save_configs(APP_CONFIGS)
            break
        p_values = values
        if '__TIMEOUT__' in event:
            if values['toggle_progressbar']:
                window['_star_bar1_'].UpdateBar(next(next_val_gen))
                window['_star_bar2_'].UpdateBar(next(next_val_gen))
        if '__TIMEOUT__' not in event:
            print('PSG event>', event)
        if event == 'toggle_progressbar':
            my_timeout = None if not values['toggle_progressbar'] else 100
        if event == '-README-listbox-':
            metadata = values['-README-listbox-'][0].log_obj['message_metadata']
            print(f'metadata = {metadata}')
        if event == '-CALL_REF-listbox-':
            ParsingError_obj = values['-CALL_REF-listbox-'][0]
            metadata = ParsingError_obj.log_obj['message_metadata']
            if 'lineno' in metadata.keys():
                lineno = metadata['lineno']
                texteditor = values['_text_editor_combo_']
                psg_module_path_SDK = psg_module_path.replace('__init__.py', 'PySimpleGUI.py')
                if 'pycharm' == texteditor:
                    texteditor = values['_PyCharm_path_']
                    subprocess.Popen(f'"{texteditor}" --line {lineno} "{psg_module_path_SDK}"', shell=True)
                elif 'subl' == texteditor:
                    subprocess.Popen(f'{texteditor} "{psg_module_path_SDK}:{lineno}"', shell=True)
        if event == '-run-' or 'F1' in event:
            update_compilation_in_psg(values)
        if event == '-open_db_folder-':
            opendir(cd)
        if event == 'open in explorer_readme':
            opendir(os.path.dirname(os.path.join(cd, values['README_OFILE'])))
        if event == 'open in explorer_calref':
            opendir(os.path.dirname(os.path.join(cd, values['CALL_REF_OFILE'])))
        if event == 'open file - readme':
            openfile(os.path.join(cd, values['README_OFILE']))
        if event == 'open file - calref':
            openfile(os.path.join(cd, values['CALL_REF_OFILE']))
        if event == 'open_init_file':
            openfile(psg_module_path)
        if event == 'open_psg_file':
            openfile(psg_module_path.replace('__init__.py', 'PySimpleGUI.py'))
        if 'F2' in event:
            window['show_time'](not values['show_time'])
        if 'F3' in event:
            window['use_psg_color'](not values['use_psg_color'])
    window.close()
if __name__ == '__main__':
    mini_GUI()