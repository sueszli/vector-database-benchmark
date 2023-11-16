"""
Function:
    炮塔类
Author:
    Charles
微信公众号:
    Charles的皮卡丘
"""
import pygame
from .arrow import Arrow
'炮塔类'

class Turret(pygame.sprite.Sprite):

    def __init__(self, turret_type, cfg, resource_loader):
        if False:
            for i in range(10):
                print('nop')
        assert turret_type in range(3)
        pygame.sprite.Sprite.__init__(self)
        self.cfg = cfg
        self.turret_type = turret_type
        self.resource_loader = resource_loader
        self.images = [resource_loader.images['game']['basic_tower'], resource_loader.images['game']['med_tower'], resource_loader.images['game']['heavy_tower']]
        self.image = self.images[turret_type]
        self.rect = self.image.get_rect()
        self.arrow = Arrow(turret_type, cfg, resource_loader)
        self.coord = (0, 0)
        self.position = (0, 0)
        (self.rect.left, self.rect.top) = self.position
        self.reset()
    '射击'

    def shot(self, position, angle=None):
        if False:
            i = 10
            return i + 15
        arrow = None
        if not self.is_cooling:
            arrow = Arrow(self.turret_type, self.cfg, self.resource_loader)
            arrow.reset(position, angle)
            self.is_cooling = True
        if self.is_cooling:
            self.cool_time -= 1
            if self.cool_time == 0:
                self.reset()
        return arrow
    '重置'

    def reset(self):
        if False:
            return 10
        if self.turret_type == 0:
            self.price = 500
            self.cool_time = 30
            self.is_cooling = False
        elif self.turret_type == 1:
            self.price = 1000
            self.cool_time = 50
            self.is_cooling = False
        elif self.turret_type == 2:
            self.price = 1500
            self.cool_time = 100
            self.is_cooling = False