import PySimpleGUI as sg
sg.SetOptions(background_color='Grey', element_background_color='Grey', text_element_background_color='Grey', font=('Calibri', 14, 'bold'))
layout = [[sg.Text('Search Demo', font=('Calibri', 18, 'bold')), sg.ReadButton('Show Names')], [sg.Text('', size=(14, 11), relief=sg.RELIEF_SOLID, font=('Calibri', 12), background_color='White', key='_display1_'), sg.Text('', size=(14, 11), relief=sg.RELIEF_SOLID, font=('Calibri', 12), background_color='White', key='_display2_')], [sg.Text('_' * 35, font=('Calibri', 16))], [sg.InputText(size=(10, 1), key='_linear_'), sg.InputText(size=(11, 1), key='_binary_')], [sg.ReadButton('Linear Search', size=(11, 1)), sg.ReadButton('Binary Search', size=(11, 1))]]
window = sg.Window('Search Demo').Layout(layout)
names = ['Roberta', 'Kylie', 'Jenny', 'Helen', 'Andrea', 'Meredith', 'Deborah', 'Pauline', 'Belinda', 'Wendy']
sorted_names = ['Andrea', 'Belinda', 'Deborah', 'Helen', 'Jenny', 'Kylie', 'Meredith', 'Pauline', 'Roberta', 'Wendy']

def display_list(list, display):
    if False:
        for i in range(10):
            print('nop')
    names = ''
    for l in list:
        names = names + l + '\n'
    window.FindElement(display).Update(names)

def linear_search():
    if False:
        for i in range(10):
            print('nop')
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
        while True:
            i = 10
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
        if button == 'Linear Search':
            linear_search()
        if button == 'Binary Search':
            binary_search()
    else:
        break