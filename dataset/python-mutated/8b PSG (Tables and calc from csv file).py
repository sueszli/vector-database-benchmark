import sys
import PySimpleGUI as sg
import csv
sg.SetOptions(background_color='LightBlue', element_background_color='LightBlue')

def calc_ladder():
    if False:
        while True:
            i = 10
    filename = sg.PopupGetFile('Get required file', no_window=True, file_types=(('CSV Files', '*.csv'),))
    data = []
    header_list = []
    with open(filename, 'r') as infile:
        reader = csv.reader(infile)
        for i in range(1):
            header = next(reader)
            data = list(reader)
    header = header + ['%', 'Pts']
    for i in range(len(data)):
        percent = str('{:.2f}'.format(int(data[i][5]) / int(data[i][6]) * 100))
        data[i] = data[i] + [percent]
        pts = int(data[i][2]) * 4 + int(data[i][4]) * 2
        data[i] = data[i] + [pts]
    col_layout = [[sg.Table(values=data, headings=header, col_widths=(16, 4, 4, 4, 4, 6, 6, 7, 4), auto_size_columns=False, max_col_width=30, justification='right', size=(None, len(data)))]]
    layout = [[sg.Column(col_layout, size=(520, 360), scrollable=True)]]
    window = sg.Window('AFL Ladder', location=(500, 310), grab_anywhere=False).Layout(layout)
    (b, v) = window.Read()
slayout = [[sg.Text('Load AFL file to display results with points and percentage', font=('Arial', 14, 'bold')), sg.ReadButton('Load File', font=('Arial', 14, 'bold'), size=(15, 1))]]
swindow = sg.Window('Load File', location=(500, 250)).Layout(slayout)
while True:
    (button, value) = swindow.Read()
    if button is not None:
        if button == 'Load File':
            calc_ladder()
    else:
        break