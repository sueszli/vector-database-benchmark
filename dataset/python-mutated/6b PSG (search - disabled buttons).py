import PySimpleGUI as sg
sg.SetOptions(background_color='LightBlue', element_background_color='LightBlue', text_element_background_color='LightBlue', font=('Calibri', 14, 'bold'))
layout = [[sg.Text('Search Demo', font=('Calibri', 18, 'bold')), sg.ReadButton('Show Names')], [sg.Text('', size=(14, 11), relief=sg.RELIEF_SOLID, font=('Calibri', 12), background_color='White', key='_display1_'), sg.Text('', size=(14, 11), relief=sg.RELIEF_SOLID, font=('Calibri', 12), background_color='White', key='_display2_')], [sg.Text('_' * 35, font=('Calibri', 16))], [sg.InputText(size=(10, 1), key='_linear_'), sg.InputText(size=(11, 1), key='_binary_')], [sg.ReadButton('Linear Search', key='_ls_', size=(11, 1)), sg.ReadButton('Binary Search', key='_bs_', size=(11, 1))]]
window = sg.Window('Search Demo').Layout(layout)
window.Finalize()
window.FindElement('_ls_').Update(disabled=True)
window.FindElement('_bs_').Update(disabled=True)
names = ['Roberta', 'Kylie', 'Jenny', 'Helen', 'Andrea', 'Meredith', 'Deborah', 'Pauline', 'Belinda', 'Wendy']
sorted_names = ['Andrea', 'Belinda', 'Deborah', 'Helen', 'Jenny', 'Kylie', 'Meredith', 'Pauline', 'Roberta', 'Wendy']

def display_list(list, display):
    if False:
        i = 10
        return i + 15
    names = ''
    for l in list:
        names = names + l + '\n'
    window.FindElement(display).Update(names)
    window.FindElement('_ls_').Update(disabled=False)
    window.FindElement('_bs_').Update(disabled=False)

def linear_search():
    if False:
        return 10
    l = names[:]
    found = False
    for l in l:
        if l == value['_linear_']:
            found = True
            window.FindElement('_display1_').Update('Linear search\n' + l + ' found.')
            break
    if not found:
        window.FindElement('_display1_').Update(value['_linear_'] + ' was \nNot found')

def binary_search():
    if False:
        for i in range(10):
            print('nop')
    l = sorted_names[:]
    lo = 0
    hi = len(l) - 1
    found = False
    while lo <= hi:
        mid = (lo + hi) // 2
        if l[mid] == value['_binary_']:
            window.FindElement('_display2_').Update('Binary search\n' + l[mid] + ' found.')
            found = True
            break
        elif l[mid] < value['_binary_']:
            lo = mid + 1
        else:
            hi = mid - 1
    if not found:
        window.FindElement('_display2_').Update(value['_binary_'] + ' was \nNot found')
while True:
    (button, value) = window.Read()
    if button is not None:
        if button == 'Show Names':
            display_list(names, '_display1_')
            display_list(sorted_names, '_display2_')
        if button == '_ls_':
            linear_search()
        if button == '_bs_':
            binary_search()
    else:
        break