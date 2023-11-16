"""
Function:
    打地鼠(Whac-A-Mole)小游戏
Author:
    Charles
微信公众号:
    Charles的皮卡丘
"""
import os
import pygame
import random
from ...utils import QuitGame
from ..base import PygameBaseGame
from .modules import Mole, Hammer, endInterface, startInterface
'配置类'

class Config:
    rootdir = os.path.split(os.path.abspath(__file__))[0]
    FPS = 100
    SCREENSIZE = (993, 477)
    TITLE = '打地鼠 —— Charles的皮卡丘'
    HOLE_POSITIONS = [(90, -20), (405, -20), (720, -20), (90, 140), (405, 140), (720, 140), (90, 290), (405, 290), (720, 290)]
    BROWN = (150, 75, 0)
    WHITE = (255, 255, 255)
    RED = (255, 0, 0)
    RECORD_PATH = os.path.join(rootdir, 'score.rec')
    BGM_PATH = os.path.join(rootdir, 'resources/audios/bgm.mp3')
    SOUND_PATHS_DICT = {'count_down': os.path.join(rootdir, 'resources/audios/count_down.wav'), 'hammering': os.path.join(rootdir, 'resources/audios/hammering.wav')}
    FONT_PATH = os.path.join(rootdir.replace('whacamole', 'base'), 'resources/fonts/Gabriola.ttf')
    IMAGE_PATHS_DICT = {'hammer': [os.path.join(rootdir, 'resources/images/hammer0.png'), os.path.join(rootdir, 'resources/images/hammer1.png')], 'begin': [os.path.join(rootdir, 'resources/images/begin.png'), os.path.join(rootdir, 'resources/images/begin1.png')], 'again': [os.path.join(rootdir, 'resources/images/again1.png'), os.path.join(rootdir, 'resources/images/again2.png')], 'background': os.path.join(rootdir, 'resources/images/background.png'), 'end': os.path.join(rootdir, 'resources/images/end.png'), 'mole': [os.path.join(rootdir, 'resources/images/mole_1.png'), os.path.join(rootdir, 'resources/images/mole_laugh1.png'), os.path.join(rootdir, 'resources/images/mole_laugh2.png'), os.path.join(rootdir, 'resources/images/mole_laugh3.png')]}
'打地鼠(Whac-A-Mole)小游戏'

class WhacAMoleGame(PygameBaseGame):
    game_type = 'whacamole'

    def __init__(self, **kwargs):
        if False:
            return 10
        self.cfg = Config
        super(WhacAMoleGame, self).__init__(config=self.cfg, **kwargs)
    '运行游戏'

    def run(self):
        if False:
            print('Hello World!')
        (screen, resource_loader, cfg) = (self.screen, self.resource_loader, self.cfg)
        while True:
            is_restart = self.GamingInterface(screen, resource_loader, cfg)
            if not is_restart:
                break
    '游戏进行界面'

    def GamingInterface(self, screen, resource_loader, cfg):
        if False:
            i = 10
            return i + 15
        resource_loader.playbgm()
        audios = resource_loader.sounds
        font = pygame.font.Font(cfg.FONT_PATH, 40)
        bg_img = resource_loader.images['background']
        startInterface(screen, resource_loader.images['begin'])
        hole_pos = random.choice(cfg.HOLE_POSITIONS)
        change_hole_event = pygame.USEREVENT
        pygame.time.set_timer(change_hole_event, 800)
        mole = Mole(resource_loader.images['mole'], hole_pos)
        hammer = Hammer(resource_loader.images['hammer'], (500, 250))
        clock = pygame.time.Clock()
        your_score = 0
        flag = False
        init_time = pygame.time.get_ticks()
        while True:
            time_remain = round((61000 - (pygame.time.get_ticks() - init_time)) / 1000.0)
            if time_remain == 40 and (not flag):
                hole_pos = random.choice(cfg.HOLE_POSITIONS)
                mole.reset()
                mole.setPosition(hole_pos)
                pygame.time.set_timer(change_hole_event, 650)
                flag = True
            elif time_remain == 20 and flag:
                hole_pos = random.choice(cfg.HOLE_POSITIONS)
                mole.reset()
                mole.setPosition(hole_pos)
                pygame.time.set_timer(change_hole_event, 500)
                flag = False
            if time_remain == 10:
                audios['count_down'].play()
            if time_remain < 0:
                break
            count_down_text = font.render('Time: ' + str(time_remain), True, cfg.WHITE)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    QuitGame()
                elif event.type == pygame.MOUSEMOTION:
                    hammer.setPosition(pygame.mouse.get_pos())
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:
                        hammer.setHammering()
                elif event.type == change_hole_event:
                    hole_pos = random.choice(cfg.HOLE_POSITIONS)
                    mole.reset()
                    mole.setPosition(hole_pos)
            if hammer.is_hammering and (not mole.is_hammer):
                is_hammer = pygame.sprite.collide_mask(hammer, mole)
                if is_hammer:
                    audios['hammering'].play()
                    mole.setBeHammered()
                    your_score += 10
            your_score_text = font.render('Score: ' + str(your_score), True, cfg.BROWN)
            screen.blit(bg_img, (0, 0))
            screen.blit(count_down_text, (875, 8))
            screen.blit(your_score_text, (800, 430))
            mole.draw(screen)
            hammer.draw(screen)
            pygame.display.flip()
            clock.tick(60)
        try:
            best_score = int(open(cfg.RECORD_PATH).read())
        except:
            best_score = 0
        if your_score > best_score:
            f = open(cfg.RECORD_PATH, 'w')
            f.write(str(your_score))
            f.close()
        score_info = {'your_score': your_score, 'best_score': best_score}
        is_restart = endInterface(screen, resource_loader.images['end'], resource_loader.images['again'], score_info, cfg.FONT_PATH, [cfg.WHITE, cfg.RED], cfg.SCREENSIZE)
        return is_restart