"""
Function:
    接金币小游戏
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
from .modules import Hero, Food, ShowEndGameInterface
'配置类'

class Config:
    rootdir = os.path.split(os.path.abspath(__file__))[0]
    TITLE = '接金币 —— Charles的皮卡丘'
    FPS = 30
    SCREENSIZE = (800, 600)
    BACKGROUND_COLOR = (0, 160, 233)
    HIGHEST_SCORE_RECORD_FILEPATH = os.path.join(rootdir, 'highest.rec')
    IMAGE_PATHS_DICT = {'gold': os.path.join(rootdir, 'resources/images/gold.png'), 'apple': os.path.join(rootdir, 'resources/images/apple.png'), 'background': os.path.join(rootdir, 'resources/images/background.jpg'), 'hero': []}
    for i in range(1, 11):
        IMAGE_PATHS_DICT['hero'].append(os.path.join(rootdir, 'resources/images/%d.png' % i))
    BGM_PATH = os.path.join(rootdir, 'resources/audios/bgm.mp3')
    SOUND_PATHS_DICT = {'get': os.path.join(rootdir, 'resources/audios/get.wav')}
    FONT_PATHS_DICT = {'default_s': {'name': os.path.join(rootdir.replace('catchcoins', 'base'), 'resources/fonts/Gabriola.ttf'), 'size': 40}, 'default_l': {'name': os.path.join(rootdir.replace('catchcoins', 'base'), 'resources/fonts/Gabriola.ttf'), 'size': 60}}
'接金币小游戏'

class CatchCoinsGame(PygameBaseGame):
    game_type = 'catchcoins'

    def __init__(self, **kwargs):
        if False:
            return 10
        self.cfg = Config
        super(CatchCoinsGame, self).__init__(config=self.cfg, **kwargs)
    '运行游戏'

    def run(self):
        if False:
            print('Hello World!')
        flag = True
        while flag:
            (screen, resource_loader, cfg) = (self.screen, self.resource_loader, self.cfg)
            (game_images, game_sounds) = (resource_loader.images, resource_loader.sounds)
            resource_loader.playbgm()
            font = resource_loader.fonts['default_s']
            hero = Hero(game_images['hero'], position=(375, 520))
            food_sprites_group = pygame.sprite.Group()
            generate_food_freq = random.randint(10, 20)
            generate_food_count = 0
            score = 0
            highest_score = 0 if not os.path.exists(cfg.HIGHEST_SCORE_RECORD_FILEPATH) else int(open(cfg.HIGHEST_SCORE_RECORD_FILEPATH).read())
            clock = pygame.time.Clock()
            while True:
                screen.fill(0)
                screen.blit(game_images['background'], (0, 0))
                countdown_text = 'Count down: ' + str((90000 - pygame.time.get_ticks()) // 60000) + ':' + str((90000 - pygame.time.get_ticks()) // 1000 % 60).zfill(2)
                countdown_text = font.render(countdown_text, True, (0, 0, 0))
                countdown_rect = countdown_text.get_rect()
                countdown_rect.topright = [cfg.SCREENSIZE[0] - 30, 5]
                screen.blit(countdown_text, countdown_rect)
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        QuitGame()
                key_pressed = pygame.key.get_pressed()
                if key_pressed[pygame.K_a] or key_pressed[pygame.K_LEFT]:
                    hero.move(cfg.SCREENSIZE, 'left')
                if key_pressed[pygame.K_d] or key_pressed[pygame.K_RIGHT]:
                    hero.move(cfg.SCREENSIZE, 'right')
                generate_food_count += 1
                if generate_food_count > generate_food_freq:
                    generate_food_freq = random.randint(10, 20)
                    generate_food_count = 0
                    food = Food(game_images, random.choice(['gold'] * 10 + ['apple']), cfg.SCREENSIZE)
                    food_sprites_group.add(food)
                for food in food_sprites_group:
                    if food.update():
                        food_sprites_group.remove(food)
                for food in food_sprites_group:
                    if pygame.sprite.collide_mask(food, hero):
                        game_sounds['get'].play()
                        food_sprites_group.remove(food)
                        score += food.score
                        if score > highest_score:
                            highest_score = score
                hero.draw(screen)
                food_sprites_group.draw(screen)
                score_text = f'Score: {score}, Highest: {highest_score}'
                score_text = font.render(score_text, True, (0, 0, 0))
                score_rect = score_text.get_rect()
                score_rect.topleft = [5, 5]
                screen.blit(score_text, score_rect)
                if pygame.time.get_ticks() >= 90000:
                    break
                pygame.display.flip()
                clock.tick(cfg.FPS)
            fp = open(cfg.HIGHEST_SCORE_RECORD_FILEPATH, 'w')
            fp.write(str(highest_score))
            fp.close()
            flag = ShowEndGameInterface(screen, cfg, score, highest_score, resource_loader)