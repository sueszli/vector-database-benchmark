"""
Function:
    炸弹人小游戏
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
from .modules import Wall, Background, Fruit, Bomb, Hero, showText, Button, Interface, mapParser
'配置类'

class Config:
    rootdir = os.path.split(os.path.abspath(__file__))[0]
    SCREENSIZE = (640, 480)
    BLOCKSIZE = 30
    FPS = 30
    YELLOW = (255, 255, 0)
    BLUE = (0, 0, 255)
    RED = (255, 0, 0)
    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)
    TITLE = '炸弹人小游戏 —— Charles的皮卡丘'
    BGM_PATH = os.path.join(rootdir, 'resources/audios/bgm.mp3')
    GAMEMAPPATHS = []
    for path in ['resources/maps/1.map', 'resources/maps/2.map']:
        GAMEMAPPATHS.append(os.path.join(rootdir, path))
    IMAGE_PATHS_DICT = {'wall': [os.path.join(rootdir, 'resources/images/misc/wall0.png'), os.path.join(rootdir, 'resources/images/misc/wall1.png'), os.path.join(rootdir, 'resources/images/misc/wall2.png')], 'dk': [os.path.join(rootdir, 'resources/images/dk/left.png'), os.path.join(rootdir, 'resources/images/dk/right.png'), os.path.join(rootdir, 'resources/images/dk/up.png'), os.path.join(rootdir, 'resources/images/dk/down.png')], 'zelda': [os.path.join(rootdir, 'resources/images/zelda/left.png'), os.path.join(rootdir, 'resources/images/zelda/right.png'), os.path.join(rootdir, 'resources/images/zelda/up.png'), os.path.join(rootdir, 'resources/images/zelda/down.png')], 'batman': [os.path.join(rootdir, 'resources/images/batman/left.png'), os.path.join(rootdir, 'resources/images/batman/right.png'), os.path.join(rootdir, 'resources/images/batman/up.png'), os.path.join(rootdir, 'resources/images/batman/down.png')], 'fruit': {'banana': os.path.join(rootdir, 'resources/images/misc/banana.png'), 'cherry': os.path.join(rootdir, 'resources/images/misc/cherry.png')}, 'background': [os.path.join(rootdir, 'resources/images/misc/bg0.png'), os.path.join(rootdir, 'resources/images/misc/bg1.png'), os.path.join(rootdir, 'resources/images/misc/bg2.png')], 'bomb': os.path.join(rootdir, 'resources/images/misc/bomb.png'), 'fire': os.path.join(rootdir, 'resources/images/misc/fire.png')}
'炸弹人小游戏'

class BomberManGame(PygameBaseGame):
    game_type = 'bomberman'

    def __init__(self, **kwargs):
        if False:
            i = 10
            return i + 15
        self.cfg = Config
        super(BomberManGame, self).__init__(config=self.cfg, **kwargs)
    '运行游戏'

    def run(self):
        if False:
            return 10
        while True:
            (screen, resource_loader, cfg) = (self.screen, self.resource_loader, self.cfg)
            resource_loader.playbgm()
            Interface(screen, cfg, mode='game_start')
            font = pygame.font.SysFont('Consolas', 15)
            for gamemap_path in cfg.GAMEMAPPATHS:
                map_parser = mapParser(gamemap_path, bg_images=resource_loader.images['background'], wall_images=resource_loader.images['wall'], blocksize=cfg.BLOCKSIZE)
                fruit_sprite_group = pygame.sprite.Group()
                used_spaces = []
                for i in range(5):
                    coordinate = map_parser.randomGetSpace(used_spaces)
                    used_spaces.append(coordinate)
                    fruit_kind = random.choice(list(resource_loader.images['fruit'].keys()))
                    fruit_sprite_group.add(Fruit(resource_loader.images['fruit'][fruit_kind], fruit_kind, coordinate=coordinate, blocksize=cfg.BLOCKSIZE))
                coordinate = map_parser.randomGetSpace(used_spaces)
                used_spaces.append(coordinate)
                ourhero = Hero(images=resource_loader.images['zelda'], coordinate=coordinate, blocksize=cfg.BLOCKSIZE, map_parser=map_parser, hero_name='ZELDA')
                aihero_sprite_group = pygame.sprite.Group()
                coordinate = map_parser.randomGetSpace(used_spaces)
                aihero_sprite_group.add(Hero(images=resource_loader.images['batman'], coordinate=coordinate, blocksize=cfg.BLOCKSIZE, map_parser=map_parser, hero_name='BATMAN'))
                used_spaces.append(coordinate)
                coordinate = map_parser.randomGetSpace(used_spaces)
                aihero_sprite_group.add(Hero(images=resource_loader.images['dk'], coordinate=coordinate, blocksize=cfg.BLOCKSIZE, map_parser=map_parser, hero_name='DK'))
                used_spaces.append(coordinate)
                bomb_sprite_group = pygame.sprite.Group()
                is_win_flag = False
                screen = pygame.display.set_mode(map_parser.screen_size)
                clock = pygame.time.Clock()
                while True:
                    dt = clock.tick(cfg.FPS)
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            QuitGame()
                        elif event.type == pygame.KEYDOWN:
                            if event.key == pygame.K_UP:
                                ourhero.move('up')
                            elif event.key == pygame.K_DOWN:
                                ourhero.move('down')
                            elif event.key == pygame.K_LEFT:
                                ourhero.move('left')
                            elif event.key == pygame.K_RIGHT:
                                ourhero.move('right')
                            elif event.key == pygame.K_SPACE:
                                if ourhero.bomb_cooling_count <= 0:
                                    bomb_sprite_group.add(ourhero.generateBomb(image=resource_loader.images['bomb'], digitalcolor=cfg.YELLOW, explode_image=resource_loader.images['fire']))
                    screen.fill(cfg.WHITE)
                    for hero in aihero_sprite_group:
                        (action, flag) = hero.randomAction(dt)
                        if flag and action == 'dropbomb':
                            bomb_sprite_group.add(hero.generateBomb(image=resource_loader.images['bomb'], digitalcolor=cfg.YELLOW, explode_image=resource_loader.images['fire']))
                    ourhero.eatFruit(fruit_sprite_group)
                    for hero in aihero_sprite_group:
                        hero.eatFruit(fruit_sprite_group)
                    map_parser.draw(screen)
                    for bomb in bomb_sprite_group:
                        if not bomb.is_being:
                            bomb_sprite_group.remove(bomb)
                        explode_area = bomb.draw(screen, dt, map_parser)
                        if explode_area:
                            if ourhero.coordinate in explode_area:
                                ourhero.health_value -= bomb.harm_value
                            for hero in aihero_sprite_group:
                                if hero.coordinate in explode_area:
                                    hero.health_value -= bomb.harm_value
                    fruit_sprite_group.draw(screen)
                    for hero in aihero_sprite_group:
                        hero.draw(screen, dt)
                    ourhero.draw(screen, dt)
                    pos_x = showText(screen, font, text=ourhero.hero_name + '(our):' + str(ourhero.health_value), color=cfg.YELLOW, position=[5, 5])
                    for hero in aihero_sprite_group:
                        (pos_x, pos_y) = (pos_x + 15, 5)
                        pos_x = showText(screen, font, text=hero.hero_name + '(ai):' + str(hero.health_value), color=cfg.YELLOW, position=[pos_x, pos_y])
                    if ourhero.health_value <= 0:
                        is_win_flag = False
                        break
                    for hero in aihero_sprite_group:
                        if hero.health_value <= 0:
                            aihero_sprite_group.remove(hero)
                    if len(aihero_sprite_group) == 0:
                        is_win_flag = True
                        break
                    pygame.display.update()
                    clock.tick(cfg.FPS)
                if is_win_flag:
                    Interface(screen, cfg, mode='game_switch')
                else:
                    break
            Interface(screen, cfg, mode='game_end')