import pygame
import PySimpleGUI as sg
import os
'\n    Demo - Simple Snake Game using PyGame and PySimpleGUI\n    This demo may not be fully functional in terms of getting the coordinate\n    systems right or other problems due to a lack of understanding of PyGame\n    The purpose of the demo is to show one way of adding a PyGame window into your PySimpleGUI window\n    Note, you must click on the game area in order for PyGame to get keyboard strokes, etc.\n    Tried using set_focus to switch to the PyGame canvas but still needed to click on game area\n'
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
segment_width = 15
segment_height = 15
segment_margin = 3
x_change = segment_width + segment_margin
y_change = 0

class Segment(pygame.sprite.Sprite):
    """ Class to represent one segment of the snake. """

    def __init__(self, x, y):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.image = pygame.Surface([segment_width, segment_height])
        self.image.fill(WHITE)
        self.rect = self.image.get_rect()
        self.rect.x = x
        self.rect.y = y
layout = [[sg.Text('Snake Game - PySimpleGUI + PyGame')], [sg.Graph((800, 600), (0, 0), (800, 600), background_color='lightblue', key='-GRAPH-')], [sg.Exit()]]
window = sg.Window('Snake Game using PySimpleGUI and PyGame', layout, finalize=True)
graph = window['-GRAPH-']
embed = graph.TKCanvas
os.environ['SDL_WINDOWID'] = str(embed.winfo_id())
os.environ['SDL_VIDEODRIVER'] = 'windib'
screen = pygame.display.set_mode((800, 600))
screen.fill(pygame.Color(255, 255, 255))
pygame.display.init()
pygame.display.update()
pygame.display.set_caption('Snake Example')
allspriteslist = pygame.sprite.Group()
snake_segments = []
for i in range(15):
    x = 250 - (segment_width + segment_margin) * i
    y = 30
    segment = Segment(x, y)
    snake_segments.append(segment)
    allspriteslist.add(segment)
clock = pygame.time.Clock()
while True:
    (event, values) = window.read(timeout=10)
    if event in (sg.WIN_CLOSED, 'Exit'):
        break
    pygame.display.update()
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            break
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_LEFT:
                x_change = (segment_width + segment_margin) * -1
                y_change = 0
            if event.key == pygame.K_RIGHT:
                x_change = segment_width + segment_margin
                y_change = 0
            if event.key == pygame.K_UP:
                x_change = 0
                y_change = (segment_height + segment_margin) * -1
            if event.key == pygame.K_DOWN:
                x_change = 0
                y_change = segment_height + segment_margin
    old_segment = snake_segments.pop()
    allspriteslist.remove(old_segment)
    x = snake_segments[0].rect.x + x_change
    y = snake_segments[0].rect.y + y_change
    segment = Segment(x, y)
    snake_segments.insert(0, segment)
    allspriteslist.add(segment)
    screen.fill(BLACK)
    allspriteslist.draw(screen)
    pygame.display.flip()
    clock.tick(5)
window.close()