"""
Function:
    愤怒的小鸟
Author:
    Charles
微信公众号:
    Charles的皮卡丘
"""
import os
import pygame
from ...utils import QuitGame
from ..base import PygameBaseGame
from .modules import GameLevels, Pig, Bird, Block, Slingshot, Slab, Button, Label
'配置类'

class Config:
    rootdir = os.path.split(os.path.abspath(__file__))[0]
    FPS = 60
    SCREENSIZE = (1800, 700)
    TITLE = '愤怒的小鸟 —— Charles的皮卡丘'
    BACKGROUND_COLOR = (51, 51, 51)
    BGM_PATH = os.path.join(rootdir, 'resources/audios/bgm.ogg')
    IMAGE_PATHS_DICT = {'pig': [os.path.join(rootdir, 'resources/images/pig_1.png'), os.path.join(rootdir, 'resources/images/pig_2.png'), os.path.join(rootdir, 'resources/images/pig_damaged.png')], 'bird': [os.path.join(rootdir, 'resources/images/bird.png')], 'wall': [os.path.join(rootdir, 'resources/images/wall_horizontal.png'), os.path.join(rootdir, 'resources/images/wall_vertical.png')], 'block': [os.path.join(rootdir, 'resources/images/block.png'), os.path.join(rootdir, 'resources/images/block_destroyed.png')]}
    FONT_PATHS_DICT_NOINIT = {'Comic_Kings': os.path.join(rootdir, 'resources/fonts/Comic_Kings.ttf'), 'arfmoochikncheez': os.path.join(rootdir, 'resources/fonts/arfmoochikncheez.ttf')}
'愤怒的小鸟'

class AngryBirdsGame(PygameBaseGame):
    game_type = 'angrybirds'

    def __init__(self, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        self.cfg = Config
        super(AngryBirdsGame, self).__init__(config=self.cfg, **kwargs)
    '运行游戏'

    def run(self):
        if False:
            while True:
                i = 10
        (screen, resource_loader, cfg) = (self.screen, self.resource_loader, self.cfg)
        resource_loader.playbgm()

        def startgame():
            if False:
                return 10
            game_levels = GameLevels(cfg, resource_loader, screen)
            game_levels.start()
        components = pygame.sprite.Group()
        title_label = Label(screen, 700, 100, 400, 200)
        title_label.addtext('ANGRY BIRDS', 80, cfg.FONT_PATHS_DICT_NOINIT['arfmoochikncheez'], (236, 240, 241))
        components.add(title_label)
        start_btn = Button(screen, 500, 400, 300, 100, startgame, (244, 208, 63), (247, 220, 111))
        start_btn.addtext('START GAME', 60, cfg.FONT_PATHS_DICT_NOINIT['arfmoochikncheez'], cfg.BACKGROUND_COLOR)
        components.add(start_btn)
        quit_btn = Button(screen, 1000, 400, 300, 100, QuitGame, (241, 148, 138), (245, 183, 177))
        quit_btn.addtext('QUIT', 60, cfg.FONT_PATHS_DICT_NOINIT['arfmoochikncheez'], cfg.BACKGROUND_COLOR)
        components.add(quit_btn)
        charles_label = Label(screen, cfg.SCREENSIZE[0] - 300, cfg.SCREENSIZE[1] - 80, 300, 100)
        charles_label.addtext('CHARLES', 60, cfg.FONT_PATHS_DICT_NOINIT['arfmoochikncheez'], (41, 41, 41))
        components.add(charles_label)
        clock = pygame.time.Clock()
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    QuitGame()
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        QuitGame()
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if start_btn.selected():
                        start_btn.action()
                    elif quit_btn.selected():
                        quit_btn.action()
            screen.fill(cfg.BACKGROUND_COLOR)
            for component in components:
                component.draw()
            pygame.display.update()
            clock.tick(cfg.FPS)