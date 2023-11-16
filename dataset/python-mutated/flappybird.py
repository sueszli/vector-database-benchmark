"""
Function:
    飞扬的小鸟小游戏
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
from .modules import GameEndIterface, GameStartInterface, Bird, Pipe
'配置类'

class Config:
    rootdir = os.path.split(os.path.abspath(__file__))[0]
    FPS = 60
    SCREENSIZE = (288, 512)
    PIPE_GAP_SIZE = 100
    TITLE = 'Flappy Bird —— Charles的皮卡丘'
    SOUND_PATHS_DICT = {'die': os.path.join(rootdir, 'resources/audios/die.wav'), 'hit': os.path.join(rootdir, 'resources/audios/hit.wav'), 'point': os.path.join(rootdir, 'resources/audios/point.wav'), 'swoosh': os.path.join(rootdir, 'resources/audios/swoosh.wav'), 'wing': os.path.join(rootdir, 'resources/audios/wing.wav')}
    IMAGE_PATHS_DICT = {'number': {'0': os.path.join(rootdir, 'resources/images/0.png'), '1': os.path.join(rootdir, 'resources/images/1.png'), '2': os.path.join(rootdir, 'resources/images/2.png'), '3': os.path.join(rootdir, 'resources/images/3.png'), '4': os.path.join(rootdir, 'resources/images/4.png'), '5': os.path.join(rootdir, 'resources/images/5.png'), '6': os.path.join(rootdir, 'resources/images/6.png'), '7': os.path.join(rootdir, 'resources/images/7.png'), '8': os.path.join(rootdir, 'resources/images/8.png'), '9': os.path.join(rootdir, 'resources/images/9.png')}, 'bird': {'red': {'up': os.path.join(rootdir, 'resources/images/redbird-upflap.png'), 'mid': os.path.join(rootdir, 'resources/images/redbird-midflap.png'), 'down': os.path.join(rootdir, 'resources/images/redbird-downflap.png')}, 'blue': {'up': os.path.join(rootdir, 'resources/images/bluebird-upflap.png'), 'mid': os.path.join(rootdir, 'resources/images/bluebird-midflap.png'), 'down': os.path.join(rootdir, 'resources/images/bluebird-downflap.png')}, 'yellow': {'up': os.path.join(rootdir, 'resources/images/yellowbird-upflap.png'), 'mid': os.path.join(rootdir, 'resources/images/yellowbird-midflap.png'), 'down': os.path.join(rootdir, 'resources/images/yellowbird-downflap.png')}}, 'background': {'day': os.path.join(rootdir, 'resources/images/background-day.png'), 'night': os.path.join(rootdir, 'resources/images/background-night.png')}, 'pipe': {'green': os.path.join(rootdir, 'resources/images/pipe-green.png'), 'red': os.path.join(rootdir, 'resources/images/pipe-red.png')}, 'others': {'gameover': os.path.join(rootdir, 'resources/images/gameover.png'), 'message': os.path.join(rootdir, 'resources/images/message.png'), 'base': os.path.join(rootdir, 'resources/images/base.png')}}
'飞扬的小鸟小游戏'

class FlappyBirdGame(PygameBaseGame):
    game_type = 'flappybird'

    def __init__(self, **kwargs):
        if False:
            print('Hello World!')
        self.cfg = Config
        super(FlappyBirdGame, self).__init__(config=self.cfg, **kwargs)
    '运行游戏'

    def run(self):
        if False:
            while True:
                i = 10
        while True:
            (screen, resource_loader, cfg) = (self.screen, self.resource_loader, self.cfg)
            sounds = resource_loader.sounds
            number_images = resource_loader.images['number']
            for key in number_images:
                number_images[key] = number_images[key].convert_alpha()
            pipe_images = dict()
            pipe_images['bottom'] = random.choice(list(resource_loader.images['pipe'].values())).convert_alpha()
            pipe_images['top'] = pygame.transform.rotate(pipe_images['bottom'], 180)
            bird_images = random.choice(list(resource_loader.images['bird'].values()))
            for key in bird_images:
                bird_images[key] = bird_images[key].convert_alpha()
            backgroud_image = random.choice(list(resource_loader.images['background'].values())).convert_alpha()
            other_images = resource_loader.images['others']
            for key in other_images:
                other_images[key] = other_images[key].convert_alpha()
            game_start_info = GameStartInterface(screen, sounds, bird_images, other_images, backgroud_image, cfg)
            score = 0
            (bird_pos, base_pos, bird_idx) = list(game_start_info.values())
            base_diff_bg = other_images['base'].get_width() - backgroud_image.get_width()
            clock = pygame.time.Clock()
            pipe_sprites = pygame.sprite.Group()
            for i in range(2):
                pipe_pos = Pipe.randomPipe(cfg, pipe_images.get('top'))
                pipe_sprites.add(Pipe(image=pipe_images.get('top'), position=(cfg.SCREENSIZE[0] + 200 + i * cfg.SCREENSIZE[0] / 2, pipe_pos.get('top')[-1])))
                pipe_sprites.add(Pipe(image=pipe_images.get('bottom'), position=(cfg.SCREENSIZE[0] + 200 + i * cfg.SCREENSIZE[0] / 2, pipe_pos.get('bottom')[-1])))
            bird = Bird(images=bird_images, idx=bird_idx, position=bird_pos)
            is_add_pipe = True
            is_game_running = True
            while is_game_running:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                        QuitGame()
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_SPACE or event.key == pygame.K_UP:
                            bird.setFlapped()
                            sounds['wing'].play()
                for pipe in pipe_sprites:
                    if pygame.sprite.collide_mask(bird, pipe):
                        sounds['hit'].play()
                        is_game_running = False
                boundary_values = [0, base_pos[-1]]
                is_dead = bird.update(boundary_values, float(clock.tick(cfg.FPS)) / 1000.0)
                if is_dead:
                    sounds['hit'].play()
                    is_game_running = False
                base_pos[0] = -((-base_pos[0] + 4) % base_diff_bg)
                flag = False
                for pipe in pipe_sprites:
                    pipe.rect.left -= 4
                    if pipe.rect.centerx < bird.rect.centerx and (not pipe.used_for_score):
                        pipe.used_for_score = True
                        score += 0.5
                        if '.5' in str(score):
                            sounds['point'].play()
                    if pipe.rect.left < 5 and pipe.rect.left > 0 and is_add_pipe:
                        pipe_pos = Pipe.randomPipe(cfg, pipe_images.get('top'))
                        pipe_sprites.add(Pipe(image=pipe_images.get('top'), position=pipe_pos.get('top')))
                        pipe_sprites.add(Pipe(image=pipe_images.get('bottom'), position=pipe_pos.get('bottom')))
                        is_add_pipe = False
                    elif pipe.rect.right < 0:
                        pipe_sprites.remove(pipe)
                        flag = True
                if flag:
                    is_add_pipe = True
                screen.blit(backgroud_image, (0, 0))
                pipe_sprites.draw(screen)
                screen.blit(other_images['base'], base_pos)
                self.showScore(cfg, screen, score, number_images)
                bird.draw(screen)
                pygame.display.update()
                clock.tick(cfg.FPS)
            GameEndIterface(screen, sounds, self.showScore, score, number_images, bird, pipe_sprites, backgroud_image, other_images, base_pos, cfg)
    '显示当前分数'

    @staticmethod
    def showScore(cfg, screen, score, number_images):
        if False:
            i = 10
            return i + 15
        digits = list(str(int(score)))
        width = 0
        for d in digits:
            width += number_images.get(d).get_width()
        offset = (cfg.SCREENSIZE[0] - width) / 2
        for d in digits:
            screen.blit(number_images.get(d), (offset, cfg.SCREENSIZE[1] * 0.1))
            offset += number_images.get(d).get_width()