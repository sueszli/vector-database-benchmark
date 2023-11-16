"""
Function:
    滑雪小游戏
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
'配置类'

class Config:
    rootdir = os.path.split(os.path.abspath(__file__))[0]
    FPS = 40
    SCREENSIZE = (640, 640)
    TITLE = '滑雪游戏 —— Charles的皮卡丘'
    IMAGE_PATHS_DICT = {'skier': [os.path.join(rootdir, 'resources/images/skier_forward.png'), os.path.join(rootdir, 'resources/images/skier_right1.png'), os.path.join(rootdir, 'resources/images/skier_right2.png'), os.path.join(rootdir, 'resources/images/skier_left2.png'), os.path.join(rootdir, 'resources/images/skier_left1.png'), os.path.join(rootdir, 'resources/images/skier_fall.png')], 'tree': os.path.join(rootdir, 'resources/images/tree.png'), 'flag': os.path.join(rootdir, 'resources/images/flag.png')}
    BGM_PATH = os.path.join(rootdir, 'resources/audios/bgm.mp3')
    FONT_PATHS_DICT = {'1/5screenwidth': {'name': os.path.join(rootdir.replace('ski', 'base'), 'resources/fonts/simkai.ttf'), 'size': SCREENSIZE[0] // 5}, '1/20screenwidth': {'name': os.path.join(rootdir.replace('ski', 'base'), 'resources/fonts/simkai.ttf'), 'size': SCREENSIZE[0] // 20}, 'default': {'name': os.path.join(rootdir.replace('ski', 'base'), 'resources/fonts/simkai.ttf'), 'size': 30}}
'滑雪者类'

class SkierSprite(pygame.sprite.Sprite):

    def __init__(self, images):
        if False:
            print('Hello World!')
        pygame.sprite.Sprite.__init__(self)
        self.direction = 0
        self.image_fall = images[-1]
        self.images = images[:-1]
        self.image = self.images[self.direction]
        self.rect = self.image.get_rect()
        self.rect.center = [320, 100]
        self.speed = [self.direction, 6 - abs(self.direction) * 2]
    '改变滑雪者的朝向. 负数为向左，正数为向右，0为向前'

    def turn(self, num):
        if False:
            i = 10
            return i + 15
        self.direction += num
        self.direction = max(-2, self.direction)
        self.direction = min(2, self.direction)
        center = self.rect.center
        self.image = self.images[self.direction]
        self.rect = self.image.get_rect()
        self.rect.center = center
        self.speed = [self.direction, 6 - abs(self.direction) * 2]
        return self.speed
    '移动滑雪者'

    def move(self):
        if False:
            return 10
        self.rect.centerx += self.speed[0]
        self.rect.centerx = max(20, self.rect.centerx)
        self.rect.centerx = min(620, self.rect.centerx)
    '设置为摔倒状态'

    def setFall(self):
        if False:
            return 10
        self.image = self.image_fall
    '设置为站立状态'

    def setForward(self):
        if False:
            print('Hello World!')
        self.direction = 0
        self.image = self.images[self.direction]
'障碍物类'

class ObstacleSprite(pygame.sprite.Sprite):

    def __init__(self, image, location, attribute):
        if False:
            i = 10
            return i + 15
        pygame.sprite.Sprite.__init__(self)
        self.image = image
        self.location = location
        self.rect = self.image.get_rect()
        self.rect.center = self.location
        self.attribute = attribute
        self.passed = False
    '移动'

    def move(self, num):
        if False:
            for i in range(10):
                print('nop')
        self.rect.centery = self.location[1] - num
'滑雪小游戏'

class SkiGame(PygameBaseGame):
    game_type = 'ski'

    def __init__(self, **kwargs):
        if False:
            return 10
        self.cfg = Config()
        super(SkiGame, self).__init__(config=self.cfg, **kwargs)
    '运行游戏'

    def run(self):
        if False:
            for i in range(10):
                print('nop')
        (screen, resource_loader, cfg) = (self.screen, self.resource_loader, self.cfg)
        resource_loader.playbgm()
        self.ShowStartInterface(screen)
        skier = SkierSprite(resource_loader.images['skier'])
        obstacles0 = self.createObstacles(20, 29)
        obstacles1 = self.createObstacles(10, 19)
        obstaclesflag = 0
        obstacles = self.AddObstacles(obstacles0, obstacles1)
        clock = pygame.time.Clock()
        distance = 0
        score = 0
        speed = [0, 6]
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    QuitGame()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_LEFT or event.key == pygame.K_a:
                        speed = skier.turn(-1)
                    elif event.key == pygame.K_RIGHT or event.key == pygame.K_d:
                        speed = skier.turn(1)
            skier.move()
            distance += speed[1]
            if distance >= 640 and obstaclesflag == 0:
                obstaclesflag = 1
                obstacles0 = self.createObstacles(20, 29)
                obstacles = self.AddObstacles(obstacles0, obstacles1)
            if distance >= 1280 and obstaclesflag == 1:
                obstaclesflag = 0
                distance -= 1280
                for obstacle in obstacles0:
                    obstacle.location[1] = obstacle.location[1] - 1280
                obstacles1 = self.createObstacles(10, 19)
                obstacles = self.AddObstacles(obstacles0, obstacles1)
            for obstacle in obstacles:
                obstacle.move(distance)
            hitted_obstacles = pygame.sprite.spritecollide(skier, obstacles, False)
            if hitted_obstacles:
                if hitted_obstacles[0].attribute == 'tree' and (not hitted_obstacles[0].passed):
                    score -= 50
                    skier.setFall()
                    self.updateFrame(screen, obstacles, skier, score)
                    pygame.time.delay(1000)
                    skier.setForward()
                    speed = [0, 6]
                    hitted_obstacles[0].passed = True
                elif hitted_obstacles[0].attribute == 'flag' and (not hitted_obstacles[0].passed):
                    score += 10
                    obstacles.remove(hitted_obstacles[0])
            self.updateFrame(screen, obstacles, skier, score)
            clock.tick(cfg.FPS)
    '创建障碍物'

    def createObstacles(self, s, e, num=10):
        if False:
            return 10
        obstacles = pygame.sprite.Group()
        locations = []
        for i in range(num):
            row = random.randint(s, e)
            col = random.randint(0, 9)
            location = [col * 64 + 20, row * 64 + 20]
            if location not in locations:
                locations.append(location)
                attribute = random.choice(['tree', 'flag'])
                image = self.resource_loader.images[attribute]
                obstacle = ObstacleSprite(image, location, attribute)
                obstacles.add(obstacle)
        return obstacles
    '合并障碍物'

    def AddObstacles(self, obstacles0, obstacles1):
        if False:
            i = 10
            return i + 15
        obstacles = pygame.sprite.Group()
        for obstacle in obstacles0:
            obstacles.add(obstacle)
        for obstacle in obstacles1:
            obstacles.add(obstacle)
        return obstacles
    '显示游戏开始界面'

    def ShowStartInterface(self, screen):
        if False:
            print('Hello World!')
        screen.fill((255, 255, 255))
        tfont = self.resource_loader.fonts['1/5screenwidth']
        cfont = self.resource_loader.fonts['1/20screenwidth']
        title = tfont.render(u'滑雪游戏', True, (255, 0, 0))
        content = cfont.render(u'按任意键开始游戏', True, (0, 0, 255))
        trect = title.get_rect()
        trect.midtop = (self.cfg.SCREENSIZE[0] / 2, self.cfg.SCREENSIZE[1] / 5)
        crect = content.get_rect()
        crect.midtop = (self.cfg.SCREENSIZE[0] / 2, self.cfg.SCREENSIZE[1] / 2)
        screen.blit(title, trect)
        screen.blit(content, crect)
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    QuitGame()
                elif event.type == pygame.KEYDOWN:
                    return
            pygame.display.update()
    '显示分数'

    def showScore(self, screen, score, pos=(10, 10)):
        if False:
            return 10
        font = self.resource_loader.fonts['default']
        score_text = font.render('Score: %s' % score, True, (0, 0, 0))
        screen.blit(score_text, pos)
    '更新当前帧的游戏画面'

    def updateFrame(self, screen, obstacles, skier, score):
        if False:
            return 10
        screen.fill((255, 255, 255))
        obstacles.draw(screen)
        screen.blit(skier.image, skier.rect)
        self.showScore(screen, score)
        pygame.display.update()