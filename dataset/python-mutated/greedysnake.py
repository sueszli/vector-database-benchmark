"""
Function:
    贪吃蛇小游戏
Author:
    Charles
微信公众号:
    Charles的皮卡丘
"""
import os
import pygame
from ...utils import QuitGame
from ..base import PygameBaseGame
from .modules import drawGameGrid, showScore, EndInterface, Apple, Snake
'配置类'

class Config:
    rootdir = os.path.split(os.path.abspath(__file__))[0]
    FPS = 5
    SCREENSIZE = (800, 500)
    TITLE = '贪吃蛇小游戏 —— Charles的皮卡丘'
    BLOCK_SIZE = 20
    BLACK = (0, 0, 0)
    GAME_MATRIX_SIZE = (int(SCREENSIZE[0] / BLOCK_SIZE), int(SCREENSIZE[1] / BLOCK_SIZE))
    BGM_PATH = os.path.join(rootdir, 'resources/audios/bgm.mp3')
    FONT_PATHS_DICT = {'default30': {'name': os.path.join(rootdir.replace('greedysnake', 'base'), 'resources/fonts/Gabriola.ttf'), 'size': 30}, 'default60': {'name': os.path.join(rootdir.replace('greedysnake', 'base'), 'resources/fonts/Gabriola.ttf'), 'size': 60}}
'贪吃蛇小游戏'

class GreedySnakeGame(PygameBaseGame):
    game_type = 'greedysnake'

    def __init__(self, **kwargs):
        if False:
            print('Hello World!')
        self.cfg = Config
        super(GreedySnakeGame, self).__init__(config=self.cfg, **kwargs)
    '运行游戏'

    def run(self):
        if False:
            return 10
        (screen, resource_loader, cfg) = (self.screen, self.resource_loader, self.cfg)
        while True:
            if not self.GamingInterface(screen, resource_loader, cfg):
                break
    '游戏运行界面'

    def GamingInterface(self, screen, resource_loader, cfg):
        if False:
            i = 10
            return i + 15
        resource_loader.playbgm()
        snake = Snake(cfg)
        apple = Apple(cfg, snake.coords)
        score = 0
        clock = pygame.time.Clock()
        while True:
            screen.fill(cfg.BLACK)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    QuitGame()
                elif event.type == pygame.KEYDOWN:
                    if event.key in [pygame.K_UP, pygame.K_DOWN, pygame.K_LEFT, pygame.K_RIGHT]:
                        snake.setDirection({pygame.K_UP: 'up', pygame.K_DOWN: 'down', pygame.K_LEFT: 'left', pygame.K_RIGHT: 'right'}[event.key])
            if snake.update(apple):
                apple = Apple(cfg, snake.coords)
                score += 1
            if snake.isgameover:
                break
            drawGameGrid(cfg, screen)
            snake.draw(screen)
            apple.draw(screen)
            showScore(cfg, score, screen, resource_loader)
            pygame.display.update()
            clock.tick(cfg.FPS)
        return EndInterface(screen, cfg, resource_loader)