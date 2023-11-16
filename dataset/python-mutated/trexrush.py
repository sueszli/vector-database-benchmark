"""
Function:
    仿谷歌浏览器小恐龙游戏
Author:
    Charles
微信公众号:
    Charles的皮卡丘
"""
import os
import random
import pygame
from ...utils import QuitGame
from ..base import PygameBaseGame
from .modules import Dinosaur, Cactus, Ptera, Ground, Cloud, Scoreboard, GameEndInterface, GameStartInterface
'配置类'

class Config:
    rootdir = os.path.split(os.path.abspath(__file__))[0]
    FPS = 60
    TITLE = 'T-Rex Rush —— Charles的皮卡丘'
    BACKGROUND_COLOR = (235, 235, 235)
    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)
    SCREENSIZE = (600, 150)
    IMAGE_PATHS_DICT = {'cacti': [os.path.join(rootdir, 'resources/images/cacti-big.png'), os.path.join(rootdir, 'resources/images/cacti-small.png')], 'cloud': os.path.join(rootdir, 'resources/images/cloud.png'), 'dino': [os.path.join(rootdir, 'resources/images/dino.png'), os.path.join(rootdir, 'resources/images/dino_ducking.png')], 'gameover': os.path.join(rootdir, 'resources/images/gameover.png'), 'ground': os.path.join(rootdir, 'resources/images/ground.png'), 'numbers': os.path.join(rootdir, 'resources/images/numbers.png'), 'ptera': os.path.join(rootdir, 'resources/images/ptera.png'), 'replay': os.path.join(rootdir, 'resources/images/replay.png')}
    SOUND_PATHS_DICT = {'die': os.path.join(rootdir, 'resources/audios/die.wav'), 'jump': os.path.join(rootdir, 'resources/audios/jump.wav'), 'point': os.path.join(rootdir, 'resources/audios/point.wav')}
'仿谷歌浏览器小恐龙游戏'

class TRexRushGame(PygameBaseGame):
    game_type = 'trexrush'

    def __init__(self, **kwargs):
        if False:
            return 10
        self.cfg = Config()
        super(TRexRushGame, self).__init__(config=self.cfg, **kwargs)
    '运行游戏'

    def run(self):
        if False:
            while True:
                i = 10
        (highest_score, flag) = (0, True)
        while flag:
            (screen, resource_loader, cfg) = (self.screen, self.resource_loader, self.cfg)
            GameStartInterface(screen, resource_loader.sounds, cfg, resource_loader)
            score = 0
            score_board = Scoreboard(resource_loader.images['numbers'], position=(534, 15), bg_color=cfg.BACKGROUND_COLOR)
            highest_score = highest_score
            highest_score_board = Scoreboard(resource_loader.images['numbers'], position=(435, 15), bg_color=cfg.BACKGROUND_COLOR, is_highest=True)
            dino = Dinosaur(resource_loader.images['dino'])
            ground = Ground(resource_loader.images['ground'], position=(0, cfg.SCREENSIZE[1]))
            cloud_sprites_group = pygame.sprite.Group()
            cactus_sprites_group = pygame.sprite.Group()
            ptera_sprites_group = pygame.sprite.Group()
            add_obstacle_timer = 0
            score_timer = 0
            clock = pygame.time.Clock()
            while True:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        QuitGame()
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_SPACE or event.key == pygame.K_UP:
                            dino.jump(resource_loader.sounds)
                        elif event.key == pygame.K_DOWN:
                            dino.duck()
                    elif event.type == pygame.KEYUP and event.key == pygame.K_DOWN:
                        dino.unduck()
                screen.fill(cfg.BACKGROUND_COLOR)
                if len(cloud_sprites_group) < 5 and random.randrange(0, 300) == 10:
                    cloud_sprites_group.add(Cloud(resource_loader.images['cloud'], position=(cfg.SCREENSIZE[0], random.randrange(30, 75))))
                add_obstacle_timer += 1
                if add_obstacle_timer > random.randrange(50, 150):
                    add_obstacle_timer = 0
                    random_value = random.randrange(0, 10)
                    if random_value >= 5 and random_value <= 7:
                        cactus_sprites_group.add(Cactus(resource_loader.images['cacti']))
                    else:
                        position_ys = [cfg.SCREENSIZE[1] * 0.82, cfg.SCREENSIZE[1] * 0.75, cfg.SCREENSIZE[1] * 0.6, cfg.SCREENSIZE[1] * 0.2]
                        ptera_sprites_group.add(Ptera(resource_loader.images['ptera'], position=(600, random.choice(position_ys))))
                dino.update()
                ground.update()
                cloud_sprites_group.update()
                cactus_sprites_group.update()
                ptera_sprites_group.update()
                score_timer += 1
                if score_timer > cfg.FPS // 12:
                    score_timer = 0
                    score += 1
                    score = min(score, 99999)
                    if score > highest_score:
                        highest_score = score
                    if score % 100 == 0:
                        resource_loader.sounds['point'].play()
                    if score % 1000 == 0:
                        ground.speed -= 1
                        for item in cloud_sprites_group:
                            item.speed -= 1
                        for item in cactus_sprites_group:
                            item.speed -= 1
                        for item in ptera_sprites_group:
                            item.speed -= 1
                for item in cactus_sprites_group:
                    if pygame.sprite.collide_mask(dino, item):
                        dino.die(resource_loader.sounds)
                for item in ptera_sprites_group:
                    if pygame.sprite.collide_mask(dino, item):
                        dino.die(resource_loader.sounds)
                dino.draw(screen)
                ground.draw(screen)
                cloud_sprites_group.draw(screen)
                cactus_sprites_group.draw(screen)
                ptera_sprites_group.draw(screen)
                score_board.set(score)
                highest_score_board.set(highest_score)
                score_board.draw(screen)
                highest_score_board.draw(screen)
                pygame.display.update()
                clock.tick(cfg.FPS)
                if dino.is_dead:
                    break
            flag = GameEndInterface(screen, cfg, resource_loader)