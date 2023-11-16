"""
Function:
    2048小游戏
Author:
    Charles
微信公众号:
    Charles的皮卡丘
"""
import os
import pygame
from ...utils import QuitGame
from ..base import PygameBaseGame
from .modules import getColorByNumber, drawGameMatrix, drawScore, drawGameIntro, EndInterface, Game2048
'配置类'

class Config:
    rootdir = os.path.split(os.path.abspath(__file__))[0]
    FPS = 60
    SCREENSIZE = (650, 370)
    TITLE = '2048小游戏 —— Charles的皮卡丘'
    BG_COLOR = '#92877d'
    MAX_SCORE_FILEPATH = os.path.join(rootdir, 'score')
    MARGIN_SIZE = 10
    BLOCK_SIZE = 80
    GAME_MATRIX_SIZE = (4, 4)
    BGM_PATH = os.path.join(rootdir, 'resources/audios/bgm.mp3')
    FONTPATH = os.path.join(rootdir.replace('twozerofoureight', 'base'), 'resources/fonts/Gabriola.ttf')
'2048小游戏'

class TwoZeroFourEightGame(PygameBaseGame):
    game_type = 'twozerofoureight'

    def __init__(self, **kwargs):
        if False:
            i = 10
            return i + 15
        self.cfg = Config
        super(TwoZeroFourEightGame, self).__init__(config=self.cfg, **kwargs)
    '运行游戏'

    def run(self):
        if False:
            while True:
                i = 10
        (screen, resource_loader, cfg) = (self.screen, self.resource_loader, self.cfg)
        while True:
            if not self.GamingInterface(screen, resource_loader, cfg):
                break
    '游戏运行界面'

    def GamingInterface(self, screen, resource_loader, cfg):
        if False:
            return 10
        resource_loader.playbgm()
        game_2048 = Game2048(matrix_size=cfg.GAME_MATRIX_SIZE, max_score_filepath=cfg.MAX_SCORE_FILEPATH)
        clock = pygame.time.Clock()
        is_running = True
        while is_running:
            screen.fill(pygame.Color(cfg.BG_COLOR))
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    QuitGame()
                elif event.type == pygame.KEYDOWN:
                    if event.key in [pygame.K_UP, pygame.K_DOWN, pygame.K_LEFT, pygame.K_RIGHT]:
                        game_2048.setDirection({pygame.K_UP: 'up', pygame.K_DOWN: 'down', pygame.K_LEFT: 'left', pygame.K_RIGHT: 'right'}[event.key])
            game_2048.update()
            if game_2048.isgameover:
                game_2048.saveMaxScore()
                is_running = False
            drawGameMatrix(screen, game_2048.game_matrix, cfg)
            (start_x, start_y) = drawScore(screen, game_2048.score, game_2048.max_score, cfg)
            drawGameIntro(screen, start_x, start_y, cfg)
            pygame.display.update()
            clock.tick(cfg.FPS)
        return EndInterface(screen, cfg)