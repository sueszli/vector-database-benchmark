import PySimpleGUI as sg
import psutil
import sys
"\n    Another simple Desktop Widget using PySimpleGUI\n    This time a CPU Usage indicator.  The Widget is square.  \n    The bottom section will be shaded to \n    represent the total amount CPU currently in use.\n    Uses the theme's button color for colors.\n\n    Copyright 2020 PySimpleGUI.org\n"
ALPHA = 0.5
THEME = 'Dark purple 6'
GSIZE = (160, 160)
UPDATE_FREQUENCY_MILLISECONDS = 2 * 1000

def main(location):
    if False:
        return 10
    graph = sg.Graph(GSIZE, (0, 0), GSIZE, key='-GRAPH-')
    layout = [[graph]]
    window = sg.Window('CPU Usage Widget Square', layout, location=location, no_titlebar=True, grab_anywhere=True, margins=(0, 0), element_padding=(0, 0), alpha_channel=ALPHA, finalize=True, right_click_menu=sg.MENU_RIGHT_CLICK_EDITME_VER_EXIT, enable_close_attempted_event=True)
    text_id2 = graph.draw_text(f'CPU', (GSIZE[0] // 2, GSIZE[1] // 4), font='Any 20', text_location=sg.TEXT_LOCATION_CENTER, color=sg.theme_button_color()[0])
    while True:
        cpu_percent = psutil.cpu_percent(interval=1)
        rect_height = int(GSIZE[1] * float(cpu_percent) / 100)
        rect_id = graph.draw_rectangle((0, rect_height), (GSIZE[0], 0), fill_color=sg.theme_button_color()[1], line_width=0)
        text_id1 = graph.draw_text(f'{int(cpu_percent)}%', (GSIZE[0] // 2, GSIZE[1] // 2), font='Any 40', text_location=sg.TEXT_LOCATION_CENTER, color=sg.theme_button_color()[0])
        graph.send_figure_to_back(rect_id)
        (event, values) = window.read(timeout=UPDATE_FREQUENCY_MILLISECONDS)
        if event in (sg.WIN_CLOSE_ATTEMPTED_EVENT, 'Exit'):
            sg.user_settings_set_entry('-location-', window.current_location())
            break
        if event == 'Edit Me':
            sg.execute_editor(__file__)
        elif event == 'Version':
            sg.popup_scrolled(__file__, sg.get_versions(), location=window.current_location(), keep_on_top=True, non_blocking=True)
        graph.delete_figure(rect_id)
        graph.delete_figure(text_id1)
    window.close()
if __name__ == '__main__':
    sg.theme(THEME)
    if len(sys.argv) > 1:
        location = sys.argv[1].split(',')
        location = (int(location[0]), int(location[1]))
    else:
        location = sg.user_settings_get_entry('-location-', (None, None))
    main(location)