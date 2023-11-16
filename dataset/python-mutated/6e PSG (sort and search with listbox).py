import PySimpleGUI as sg
sg.SetOptions(background_color='DarkGrey', element_background_color='DarkGrey', text_element_background_color='DarkGrey', font=('Calibri', 14, 'bold'))
column1 = [[sg.ReadButton('Original list', size=(10, 1))], [sg.ReadButton('Default sort', size=(10, 1))], [sg.ReadButton('Sort: selection', size=(10, 1))], [sg.ReadButton('Sort: quick', size=(10, 1))]]
layout = [[sg.Text('Search and Sort Demo', font=('Calibri', 20, 'bold'))], [sg.Listbox(values=[''], size=(14, 11), font=('Calibri', 12), background_color='White', key='_display_'), sg.Column(column1)], [sg.Text('_' * 38, font=('Calibri', 16))], [sg.InputText(size=(10, 1), key='_linear_'), sg.Text('  '), sg.InputText(size=(11, 1), key='_binary_')], [sg.ReadButton('Linear Search', size=(11, 1)), sg.Text(' '), sg.ReadButton('Binary Search', size=(11, 1))]]
window = sg.Window('Search and Sort Demo').Layout(layout)
names = ['Roberta', 'Kylie', 'Jenny', 'Helen', 'Andrea', 'Meredith', 'Deborah', 'Pauline', 'Belinda', 'Wendy']

def display_list(list):
    if False:
        for i in range(10):
            print('nop')
    global list_displayed
    list_displayed = list
    values = [l for l in list]
    window.FindElement('_display_').Update(values)

def default(names):
    if False:
        return 10
    l = names[:]
    l.sort()
    display_list(l)

def sel_sort(names):
    if False:
        while True:
            i = 10
    l = names[:]
    for i in range(len(l)):
        smallest = i
        for j in range(i + 1, len(l)):
            if l[j] < l[smallest]:
                smallest = j
        (l[smallest], l[i]) = (l[i], l[smallest])
    display_list(l)

def qsort_holder(names):
    if False:
        i = 10
        return i + 15
    l = names[:]
    quick_sort(l, 0, len(l) - 1)
    display_list(l)

def quick_sort(l, first, last):
    if False:
        print('Hello World!')
    if first >= last:
        return l
    pivot = l[first]
    low = first
    high = last
    while low < high:
        while l[high] > pivot:
            high = high - 1
        while l[low] < pivot:
            low = low + 1
        if low <= high:
            (l[high], l[low]) = (l[low], l[high])
            low = low + 1
            high = high - 1
    quick_sort(l, first, low - 1)
    quick_sort(l, low, last)

def linear_search():
    if False:
        for i in range(10):
            print('nop')
    l = names[:]
    found = False
    for l in l:
        if l == value['_linear_']:
            found = True
            result = ['Linear search', l + ' found']
            window.FindElement('_display_').Update(result)
            break
    if not found:
        result = [value['_linear_'], 'was not found']
        window.FindElement('_display_').Update(result)

def binary_search():
    if False:
        while True:
            i = 10
    l = list_displayed[:]
    lo = 0
    hi = len(l) - 1
    found = False
    while lo <= hi:
        mid = (lo + hi) // 2
        if l[mid] == value['_binary_']:
            result = ['Binary search', l[mid] + ' found.']
            window.FindElement('_display_').Update(result)
            found = True
            break
        elif l[mid] < value['_binary_']:
            lo = mid + 1
        else:
            hi = mid - 1
    if not found:
        result = [value['_binary_'], 'was not found']
        window.FindElement('_display_').Update(result)
while True:
    (button, value) = window.Read()
    if button is not None:
        if button == 'Original list':
            display_list(names)
        if button == 'Default sort':
            default(names)
        if button == 'Sort: selection':
            sel_sort(names)
        if button == 'Sort: quick':
            qsort_holder(names)
        if button == 'Linear Search':
            linear_search()
        if button == 'Binary Search':
            binary_search()
    else:
        break