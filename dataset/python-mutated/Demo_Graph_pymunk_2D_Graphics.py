import PySimpleGUIWeb as sg
import pymunk
import random
import socket
'\n    Demo that shows integrating PySimpleGUI with the pymunk library.  This combination\n    of PySimpleGUI and pymunk could be used to build games.\n    Note this exact same demo runs with PySimpleGUIWeb by changing the import statement\n'

class Ball:

    def __init__(self, x, y, r, *args, **kwargs):
        if False:
            return 10
        mass = 10
        self.body = pymunk.Body(mass, pymunk.moment_for_circle(mass, 0, r, (0, 0)))
        self.body.position = (x, y)
        self.shape = pymunk.Circle(self.body, r, offset=(0, 0))
        self.shape.elasticity = 0.99999
        self.shape.friction = 0.8
        self.gui_circle_figure = None

class Playfield:

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.space = pymunk.Space()
        self.space.gravity = (0, 200)
        self.add_wall(self.space, (0, 400), (600, 400))
        self.add_wall(self.space, (0, 0), (0, 600))
        self.add_wall(self.space, (600, 0), (600, 400))

    def add_wall(self, space, pt_from, pt_to):
        if False:
            print('Hello World!')
        body = pymunk.Body(body_type=pymunk.Body.STATIC)
        ground_shape = pymunk.Segment(body, pt_from, pt_to, 0.0)
        ground_shape.friction = 0.8
        ground_shape.elasticity = 0.99
        self.space.add(ground_shape)

    def add_balls(self):
        if False:
            for i in range(10):
                print('nop')
        self.arena_balls = []
        for i in range(1, 200):
            x = random.randint(0, 600)
            y = random.randint(0, 400)
            r = random.randint(1, 10)
            ball = Ball(x, y, r)
            self.arena_balls.append(ball)
            area.space.add(ball.body, ball.shape)
            ball.gui_circle_figure = graph_elem.draw_circle((x, y), r, fill_color='black', line_color='red')
graph_elem = sg.Graph((600, 400), (0, 400), (600, 0), enable_events=True, key='_GRAPH_', background_color='lightblue')
layout = [[sg.Text('Ball Test'), sg.Text('My IP {}'.format(socket.gethostbyname(socket.gethostname())))], [graph_elem], [sg.Button('Kick'), sg.Button('Exit')]]
window = sg.Window('Window Title', layout, finalize=True)
area = Playfield()
area.add_balls()
while True:
    (event, values) = window.read(timeout=0)
    if event in (None, 'Exit'):
        break
    area.space.step(0.02)
    for ball in area.arena_balls:
        if event == 'Kick':
            ball.body.position = (ball.body.position[0], ball.body.position[1] - random.randint(1, 200))
        graph_elem.relocate_figure(ball.gui_circle_figure, ball.body.position[0], ball.body.position[1])
window.close()