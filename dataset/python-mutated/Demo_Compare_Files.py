import PySimpleGUI as sg
'\n    Simple "diff" in PySimpleGUI\n'
sg.theme('Dark Blue 3')

def GetFilesToCompare():
    if False:
        print('Hello World!')
    form_rows = [[sg.Text('Enter 2 files to comare')], [sg.Text('File 1', size=(15, 1)), sg.InputText(key='-file1-'), sg.FileBrowse()], [sg.Text('File 2', size=(15, 1)), sg.InputText(key='-file2-'), sg.FileBrowse(target='-file2-')], [sg.Submit(), sg.Cancel()]]
    window = sg.Window('File Compare', form_rows)
    (event, values) = window.read()
    window.close()
    return (event, values)

def main():
    if False:
        for i in range(10):
            print('nop')
    (button, values) = GetFilesToCompare()
    (f1, f2) = (values['-file1-'], values['-file2-'])
    if any((button != 'Submit', f1 == '', f2 == '')):
        sg.popup_error('Operation cancelled')
        return
    with open(f1, 'rb') as file1:
        with open(f2, 'rb') as file2:
            a = file1.read()
            b = file2.read()
        for (i, x) in enumerate(a):
            if x != b[i]:
                sg.popup('Compare results for files', f1, f2, '**** Mismatch at offset {} ****'.format(i))
                break
        else:
            if len(a) == len(b):
                sg.popup('**** The files are IDENTICAL ****')
if __name__ == '__main__':
    main()