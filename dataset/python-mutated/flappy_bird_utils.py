import pygame
import sys

def load():
    if False:
        i = 10
        return i + 15
    PLAYER_PATH = ('assets/sprites/newbird-upflap.png', 'assets/sprites/newbird-midflap.png', 'assets/sprites/newbird-downflap.png')
    BACKGROUND_PATH = 'assets/sprites/background-black.png'
    PIPE_PATH = 'assets/sprites/cactus-green.png'
    (IMAGES, SOUNDS, HITMASKS) = ({}, {}, {})
    IMAGES['base'] = pygame.image.load('assets/sprites/base.png').convert_alpha()
    if 'win' in sys.platform:
        soundExt = '.wav'
    else:
        soundExt = '.ogg'
    IMAGES['background'] = pygame.image.load(BACKGROUND_PATH).convert()
    IMAGES['player'] = (pygame.image.load(PLAYER_PATH[0]).convert_alpha(), pygame.image.load(PLAYER_PATH[1]).convert_alpha(), pygame.image.load(PLAYER_PATH[2]).convert_alpha())
    IMAGES['pipe'] = (pygame.transform.rotate(pygame.image.load(PIPE_PATH).convert_alpha(), 180), pygame.image.load(PIPE_PATH).convert_alpha())
    HITMASKS['pipe'] = (getHitmask(IMAGES['pipe'][0]), getHitmask(IMAGES['pipe'][1]))
    HITMASKS['player'] = (getHitmask(IMAGES['player'][0]), getHitmask(IMAGES['player'][1]), getHitmask(IMAGES['player'][2]))
    return (IMAGES, SOUNDS, HITMASKS)

def getHitmask(image):
    if False:
        for i in range(10):
            print('nop')
    "returns a hitmask using an image's alpha."
    mask = []
    for x in range(image.get_width()):
        mask.append([])
        for y in range(image.get_height()):
            mask[x].append(bool(image.get_at((x, y))[3]))
    return mask