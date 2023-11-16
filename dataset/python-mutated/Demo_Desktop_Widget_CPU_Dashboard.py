import PySimpleGUI as sg
import sys
import psutil
'\n    Desktop floating widget - CPU Cores \n    Uses psutil to display:\n        CPU usage of each individual core\n    CPU utilization is updated every 500 ms by default\n    Utiliziation is shown as a scrolling area graph\n    To achieve a "rainmeter-style" of window, these featurees were used:\n      An alpha-channel setting of 0.8 to give a little transparency\n      No titlebar\n      Grab anywhere, making window easy to move around\n    Note that the keys are tuples, with a tuple as the second item\n        (\'-KEY-\', (row, col))      \n    Copyright 2020, 2022 PySimpleGUI\n'
GRAPH_WIDTH = 120
GRAPH_HEIGHT = 40
TRANSPARENCY = 0.8
NUM_COLS = 4
POLL_FREQUENCY = 1500
colors = ('#23a0a0', '#56d856', '#be45be', '#5681d8', '#d34545', '#BE7C29')

class DashGraph(object):

    def __init__(self, graph_elem, text_elem, starting_count, color):
        if False:
            while True:
                i = 10
        self.graph_current_item = 0
        self.graph_elem = graph_elem
        self.text_elem = text_elem
        self.prev_value = starting_count
        self.max_sent = 1
        self.color = color
        self.line_list = []

    def graph_percentage_abs(self, value):
        if False:
            i = 10
            return i + 15
        self.line_list.append(self.graph_elem.draw_line((self.graph_current_item, 0), (self.graph_current_item, value), color=self.color))
        if self.graph_current_item >= GRAPH_WIDTH:
            self.graph_elem.move(-1, 0)
            self.graph_elem.delete_figure(self.line_list[0])
            self.line_list = self.line_list[1:]
        else:
            self.graph_current_item += 1

    def text_display(self, text):
        if False:
            return 10
        self.text_elem.update(text)

def main(location):
    if False:
        while True:
            i = 10

    def Txt(text, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        return sg.Text(text, font='Helvetica 8', **kwargs)

    def GraphColumn(name, key):
        if False:
            i = 10
            return i + 15
        return sg.Column([[Txt(name, size=(10, 1), key=('-TXT-', key))], [sg.Graph((GRAPH_WIDTH, GRAPH_HEIGHT), (0, 0), (GRAPH_WIDTH, 100), background_color='black', key=('-GRAPH-', key))]], pad=(2, 2))
    num_cores = len(psutil.cpu_percent(percpu=True))
    sg.theme('Black')
    layout = [[sg.Text('CPU Core Usage', justification='c', expand_x=True)]]
    for rows in range(num_cores // NUM_COLS + 1):
        layout += [[GraphColumn('CPU ' + str(rows * NUM_COLS + cols), (rows, cols)) for cols in range(min(num_cores - rows * NUM_COLS, NUM_COLS))]]
    window = sg.Window('CPU Cores Usage Widget', layout, keep_on_top=True, grab_anywhere=True, no_titlebar=True, return_keyboard_events=True, alpha_channel=TRANSPARENCY, use_default_focus=False, finalize=True, margins=(1, 1), element_padding=(0, 0), border_depth=0, location=location, enable_close_attempted_event=True, right_click_menu=sg.MENU_RIGHT_CLICK_EDITME_VER_EXIT)
    graphs = []
    for rows in range(num_cores // NUM_COLS + 1):
        for cols in range(min(num_cores - rows * NUM_COLS, NUM_COLS)):
            graphs += [DashGraph(window['-GRAPH-', (rows, cols)], window['-TXT-', (rows, cols)], 0, colors[(rows * NUM_COLS + cols) % len(colors)])]
    while True:
        (event, values) = window.read(timeout=POLL_FREQUENCY)
        if event in (sg.WIN_CLOSE_ATTEMPTED_EVENT, 'Exit'):
            sg.user_settings_set_entry('-location-', window.current_location())
            break
        elif event == sg.WIN_CLOSED:
            break
        elif event == 'Edit Me':
            sg.execute_editor(__file__)
        elif event == 'Version':
            sg.popup_scrolled(__file__, sg.get_versions(), keep_on_top=True, location=window.current_location())
        stats = psutil.cpu_percent(percpu=True)
        for (i, util) in enumerate(stats):
            graphs[i].graph_percentage_abs(util)
            graphs[i].text_display('{} CPU {:2.0f}'.format(i, util))
    window.close()
if __name__ == '__main__':
    if len(sys.argv) > 1:
        location = sys.argv[1].split(',')
        location = (int(location[0]), int(location[1]))
    else:
        location = sg.user_settings_get_entry('-location-', (None, None))
    main(location)