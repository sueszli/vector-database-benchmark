import PySimpleGUI as sg
import pymunk
import random
'\n    Demo of pymunk physics lib combined with a large Window that is transparent. Result appears like\n    a screensaver type of screen\n'

class Ball:

    def __init__(self, x, y, r, *args, **kwargs):
        if False:
            print('Hello World!')
        mass = 10
        self.body = pymunk.Body(mass, pymunk.moment_for_circle(mass, 0, r, (0, 0)))
        self.body.position = (x, y)
        self.shape = pymunk.Circle(self.body, r, offset=(0, 0))
        self.shape.elasticity = 0.99999
        self.shape.friction = 0.8
        self.gui_circle_figure = None

class Playfield:

    def __init__(self, graph_elem, screensize):
        if False:
            i = 10
            return i + 15
        self.graph_elem = graph_elem
        self.space = pymunk.Space()
        self.space.gravity = (0, 200)
        self.screensize = screensize
        self.add_wall((0, screensize[1]), (screensize[0], screensize[1]))
        self.add_wall((0, 0), (0, screensize[1]))
        self.add_wall((screensize[0], 0), (screensize[0], screensize[1]))
        self.arena_balls = []

    def add_wall(self, pt_from, pt_to):
        if False:
            for i in range(10):
                print('nop')
        body = pymunk.Body(body_type=pymunk.Body.STATIC)
        ground_shape = pymunk.Segment(body, pt_from, pt_to, 1.0)
        ground_shape.friction = 0.8
        ground_shape.elasticity = 0.99
        self.space.add(ground_shape)

    def add_balls(self, num_balls=30):
        if False:
            return 10
        for i in range(1, num_balls):
            x = random.randint(0, self.screensize[0])
            y = random.randint(0, self.screensize[1])
            r = random.randint(5, 10)
            ball = Ball(x, y, r)
            self.arena_balls.append(ball)
            self.space.add(ball.body, ball.shape)
            ball.gui_circle_figure = self.graph_elem.draw_circle((x, y), r, fill_color=random.choice(('#23a0a0', '#56d856', '#be45be', '#5681d8', '#d34545', '#BE7C29')), line_width=0)

def main():
    if False:
        print('Hello World!')
    screensize = sg.Window.get_screen_size()
    graph_elem = sg.Graph(screensize, (0, screensize[1]), (screensize[0], 0), enable_events=True, key='-GRAPH-', background_color='lightblue')
    layout = [[graph_elem]]
    window1 = sg.Window('Bouncing Balls', layout, finalize=True, location=(0, 0), keep_on_top=True, element_padding=(0, 0), margins=(0, 0), no_titlebar=True, right_click_menu=[[''], ['Front', 'Back', 'Controls', 'Exit']])
    area = Playfield(graph_elem, screensize)
    area.add_balls()
    (transparent, paused) = (False, True)
    layout2 = [[sg.B('❎', border_width=0, button_color=('white', sg.theme_background_color()), key='Exit')], [sg.B('Kick'), sg.B('Front'), sg.B('Back'), sg.B('More Balls'), sg.B('Transparent'), sg.B('Resume', key='Pause')]]
    window2 = sg.Window('Buttons', layout2, keep_on_top=True, grab_anywhere=True, no_titlebar=True, finalize=True)
    while True:
        (window, event, values) = sg.read_all_windows(timeout=0)
        if event in (sg.WIN_CLOSED, 'Exit'):
            break
        if event == 'More Balls':
            area.add_balls()
        elif event == 'Kick':
            for ball in area.arena_balls:
                ball.body.position = (ball.body.position[0], ball.body.position[1] - random.randint(200, 400))
        elif event == 'Front':
            window1.bring_to_front()
        elif event == 'Back':
            window1.send_to_back()
        elif event == 'Transparent':
            window1.set_transparent_color('lightblue' if not transparent else 'black')
            transparent = not transparent
        elif event == 'Controls':
            window2.bring_to_front()
        elif event == 'Pause':
            paused = not paused
            window['Pause'].update(text='Resume' if paused else 'Pause')
        if paused:
            continue
        area.space.step(0.02)
        for ball in area.arena_balls:
            if ball.body.position[1] > screensize[1]:
                ball.body.position = (ball.body.position[0], screensize[1] - 30)
            graph_elem.RelocateFigure(ball.gui_circle_figure, ball.body.position[0], ball.body.position[1])
    window1.close()
    window2.close()
if __name__ == '__main__':
    main()