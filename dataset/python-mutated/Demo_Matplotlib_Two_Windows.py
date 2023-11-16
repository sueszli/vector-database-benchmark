from matplotlib import use
import PySimpleGUI as sg
import matplotlib.pyplot as plt
'\n    Simultaneous PySimpleGUI Window AND a Matplotlib Interactive Window\n    A number of people have requested the ability to run a normal PySimpleGUI window that\n    launches a MatplotLib window that is interactive with the usual Matplotlib controls.\n    It turns out to be a rather simple thing to do.  The secret is to add parameter block=False to plt.show()\n'

def draw_plot():
    if False:
        print('Hello World!')
    plt.plot([0.1, 0.2, 0.5, 0.7])
    plt.show(block=False)
layout = [[sg.Button('Plot'), sg.Cancel(), sg.Button('Popup')]]
window = sg.Window('Have some Matplotlib....', layout)
while True:
    (event, values) = window.read()
    if event in (sg.WIN_CLOSED, 'Cancel'):
        break
    elif event == 'Plot':
        draw_plot()
    elif event == 'Popup':
        sg.popup('Yes, your application is still running')
window.close()