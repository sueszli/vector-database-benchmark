import PySimpleGUI
sg = PySimpleGUI
import datetime, inspect
' \n    Create All Possible Tags\n    Will output to STDOUT all of the different tags for classes, members and functions for a given PySimpleGUI.py\n    file.  Functions that begin with _ are filtered out from the list.\n    Displays the results in a PySimpleGUI window which can be used to copy and paste into other places.\n'
messages = []
log = lambda x: messages.append(x)
SHOW_functionsNmethods_that_starts_with_UNDERSCORE = True

def valid_field(pair):
    if False:
        while True:
            i = 10
    bad_fields = 'LOOK_AND_FEEL_TABLE copyright __builtins__'.split(' ')
    bad_prefix = 'TITLE_ TEXT_ ELEM_TYPE_ DEFAULT_ BUTTON_TYPE_ LISTBOX_SELECT METER_ POPUP_ THEME_'.split(' ')
    (field_name, python_object) = pair
    if type(python_object) is bytes:
        return False
    if field_name in bad_fields:
        return False
    if any([i for i in bad_prefix if field_name.startswith(i)]):
        return False
    return True
psg_members = [i for i in inspect.getmembers(PySimpleGUI) if valid_field(i)]
psg_funcs = [o[0] for o in psg_members if inspect.isfunction(o[1])]
psg_classes = [o for o in psg_members if inspect.isclass(o[1])]
psg_classes = sorted(list(set([i[1] for i in psg_classes])), key=lambda x: x.__name__)
for aclass in psg_classes:
    class_name = aclass.__name__
    if 'Tk' in class_name or 'TK' in class_name or 'Element' == class_name:
        continue
    log(f'## {class_name} Element ')
    log(f'<!-- <+{class_name}.doc+> -->')
    log(f'<!-- <+{class_name}.__init__+> -->\n')
    log('\n'.join([f'### {name}\n<!-- <+{class_name}.{name}+> -->\n' for (name, obj) in inspect.getmembers(aclass) if not name.startswith('_')]))

def get_filtered_funcs(psg_funcs, show_underscore=False):
    if False:
        i = 10
        return i + 15
    space = '-' * 30
    curr_dt = today = datetime.datetime.today()
    filtered = [f'{curr_dt}\n\n{space}Functions start here{space}\n']
    for i in psg_funcs:
        txt = f'<!-- <+func.{i}+> -->'
        if i.startswith('_') and show_underscore:
            filtered.append(txt)
            continue
        filtered.append(txt)
    return f'TOTAL funcs amount listed below: {len(filtered)}\n' + '\n'.join(filtered)
window = sg.Window('Dump of tags', [[sg.ML(size=(80, 40), key='w'), sg.Col([[sg.ML(size=(80, 40), key='w2')], [sg.CB('show _ funcs&methods?', key='-_sc-', enable_events=True)]])]], resizable=True, finalize=True)
window['w']('\n'.join(messages))
window['w2'](get_filtered_funcs(psg_funcs, True))
while True:
    (event, values) = window()
    if event in ('Exit', None):
        break
    print(event)
    if event == '-_sc-':
        filtered = get_filtered_funcs(psg_funcs, values['-_sc-'])
        window['w2'](filtered)
window.close()