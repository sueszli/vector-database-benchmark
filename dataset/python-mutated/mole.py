"""
Function:
    地鼠
Author:
    Charles
微信公众号:
    Charles的皮卡丘
"""
import pygame
'地鼠'

class Mole(pygame.sprite.Sprite):

    def __init__(self, images, position, **kwargs):
        if False:
            i = 10
            return i + 15
        pygame.sprite.Sprite.__init__(self)
        self.images = [pygame.transform.scale(images[0], (101, 103)), pygame.transform.scale(images[-1], (101, 103))]
        self.image = self.images[0]
        self.rect = self.image.get_rect()
        self.mask = pygame.mask.from_surface(self.image)
        self.setPosition(position)
        self.is_hammer = False
    '设置位置'

    def setPosition(self, pos):
        if False:
            i = 10
            return i + 15
        (self.rect.left, self.rect.top) = pos
    '设置被击中'

    def setBeHammered(self):
        if False:
            return 10
        self.is_hammer = True
    '显示在屏幕上'

    def draw(self, screen):
        if False:
            while True:
                i = 10
        if self.is_hammer:
            self.image = self.images[1]
        screen.blit(self.image, self.rect)
    '重置'

    def reset(self):
        if False:
            while True:
                i = 10
        self.image = self.images[0]
        self.is_hammer = False