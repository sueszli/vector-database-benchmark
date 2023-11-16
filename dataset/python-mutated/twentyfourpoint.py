"""
Function:
    24点小游戏
Author:
    Charles
微信公众号:
    Charles的皮卡丘
"""
import os
import pygame
from ...utils import QuitGame
from fractions import Fraction
from ..base import PygameBaseGame
from .modules import Card, Button, game24Generator
'配置类'

class Config:
    rootdir = os.path.split(os.path.abspath(__file__))[0]
    FPS = 30
    TITLE = '24点小游戏 —— Charles的皮卡丘'
    RED = (255, 0, 0)
    BLACK = (0, 0, 0)
    AZURE = (240, 255, 255)
    WHITE = (255, 255, 255)
    MISTYROSE = (255, 228, 225)
    PALETURQUOISE = (175, 238, 238)
    PAPAYAWHIP = (255, 239, 213)
    NUMBERFONT_COLORS = [BLACK, RED]
    NUMBERCARD_COLORS = [MISTYROSE, PALETURQUOISE]
    NUMBERCARD_POSITIONS = [(25, 50, 150, 200), (225, 50, 150, 200), (425, 50, 150, 200), (625, 50, 150, 200)]
    OPREATORS = ['+', '-', '×', '÷']
    OPREATORFONT_COLORS = [BLACK, RED]
    OPERATORCARD_COLORS = [MISTYROSE, PALETURQUOISE]
    OPERATORCARD_POSITIONS = [(230, 300, 50, 50), (330, 300, 50, 50), (430, 300, 50, 50), (530, 300, 50, 50)]
    BUTTONS = ['RESET', 'ANSWERS', 'NEXT']
    BUTTONFONT_COLORS = [BLACK, BLACK]
    BUTTONCARD_COLORS = [MISTYROSE, PALETURQUOISE]
    BUTTONCARD_POSITIONS = [(25, 400, 700 / 3, 150), (50 + 700 / 3, 400, 700 / 3, 150), (75 + 1400 / 3, 400, 700 / 3, 150)]
    SCREENSIZE = (800, 600)
    GROUPTYPES = ['NUMBER', 'OPREATOR', 'BUTTON']
    SOUND_PATHS_DICT = {'win': os.path.join(rootdir, 'resources/audios/win.wav'), 'lose': os.path.join(rootdir, 'resources/audios/lose.wav'), 'warn': os.path.join(rootdir, 'resources/audios/warn.wav')}
    BGM_PATH = os.path.join(rootdir.replace('twentyfourpoint', 'base'), 'resources/audios/liuyuedeyu.mp3')
    FONT_PATHS_DICT = {'default': {'name': os.path.join(rootdir.replace('twentyfourpoint', 'base'), 'resources/fonts/MaiandraGD.ttf'), 'size': 30}, 'answer': {'name': os.path.join(rootdir.replace('twentyfourpoint', 'base'), 'resources/fonts/MaiandraGD.ttf'), 'size': 20}, 'info': {'name': os.path.join(rootdir.replace('twentyfourpoint', 'base'), 'resources/fonts/MaiandraGD.ttf'), 'size': 40}, 'number': {'name': os.path.join(rootdir.replace('twentyfourpoint', 'base'), 'resources/fonts/MaiandraGD.ttf'), 'size': 50}}
'24点小游戏'

class TwentyfourPointGame(PygameBaseGame):
    game_type = 'twentyfourpoint'

    def __init__(self, **kwargs):
        if False:
            print('Hello World!')
        self.cfg = Config
        super(TwentyfourPointGame, self).__init__(config=self.cfg, **kwargs)
    '运行游戏'

    def run(self):
        if False:
            i = 10
            return i + 15
        (screen, resource_loader, cfg) = (self.screen, self.resource_loader, self.cfg)
        resource_loader.playbgm()
        (win_sound, lose_sound, warn_sound) = (resource_loader.sounds['win'], resource_loader.sounds['lose'], resource_loader.sounds['warn'])
        game24_gen = game24Generator()
        game24_gen.generate()
        number_sprites_group = self.getNumberSpritesGroup(game24_gen.numbers_now)
        operator_sprites_group = self.getOperatorSpritesGroup(cfg.OPREATORS)
        button_sprites_group = self.getButtonSpritesGroup(cfg.BUTTONS)
        clock = pygame.time.Clock()
        selected_numbers = []
        selected_operators = []
        selected_buttons = []
        is_win = False
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    QuitGame()
                elif event.type == pygame.MOUSEBUTTONUP:
                    mouse_pos = pygame.mouse.get_pos()
                    selected_numbers = self.checkClicked(number_sprites_group, mouse_pos, 'NUMBER')
                    selected_operators = self.checkClicked(operator_sprites_group, mouse_pos, 'OPREATOR')
                    selected_buttons = self.checkClicked(button_sprites_group, mouse_pos, 'BUTTON')
            screen.fill(cfg.AZURE)
            if len(selected_numbers) == 2 and len(selected_operators) == 1:
                noselected_numbers = []
                for each in number_sprites_group:
                    if each.is_selected:
                        if each.select_order == '1':
                            selected_number1 = each.attribute
                        elif each.select_order == '2':
                            selected_number2 = each.attribute
                        else:
                            raise ValueError('Unknow select_order %s, expect 1 or 2...' % each.select_order)
                    else:
                        noselected_numbers.append(each.attribute)
                    each.is_selected = False
                for each in operator_sprites_group:
                    each.is_selected = False
                result = self.calculate(selected_number1, selected_number2, *selected_operators)
                if result is not None:
                    game24_gen.numbers_now = noselected_numbers + [result]
                    is_win = game24_gen.check()
                    if is_win:
                        win_sound.play()
                    if not is_win and len(game24_gen.numbers_now) == 1:
                        lose_sound.play()
                else:
                    warn_sound.play()
                selected_numbers = []
                selected_operators = []
                number_sprites_group = self.getNumberSpritesGroup(game24_gen.numbers_now)
            for each in number_sprites_group:
                each.draw(screen, pygame.mouse.get_pos())
            for each in operator_sprites_group:
                each.draw(screen, pygame.mouse.get_pos())
            for each in button_sprites_group:
                if selected_buttons and selected_buttons[0] in ['RESET', 'NEXT']:
                    is_win = False
                if selected_buttons and each.attribute == selected_buttons[0]:
                    each.is_selected = False
                    number_sprites_group = each.do(game24_gen, self.getNumberSpritesGroup, number_sprites_group, button_sprites_group)
                    selected_buttons = []
                each.draw(screen, pygame.mouse.get_pos())
            if is_win:
                self.showInfo('Congratulations', screen)
            if not is_win and len(game24_gen.numbers_now) == 1:
                self.showInfo('Game Over', screen)
            pygame.display.flip()
            clock.tick(cfg.FPS)
    '检查控件是否被点击'

    def checkClicked(self, group, mouse_pos, group_type='NUMBER'):
        if False:
            return 10
        selected = []
        if group_type == self.cfg.GROUPTYPES[0] or group_type == self.cfg.GROUPTYPES[1]:
            max_selected = 2 if group_type == self.cfg.GROUPTYPES[0] else 1
            num_selected = 0
            for each in group:
                num_selected += int(each.is_selected)
            for each in group:
                if each.rect.collidepoint(mouse_pos):
                    if each.is_selected:
                        each.is_selected = not each.is_selected
                        num_selected -= 1
                        each.select_order = None
                    elif num_selected < max_selected:
                        each.is_selected = not each.is_selected
                        num_selected += 1
                        each.select_order = str(num_selected)
                if each.is_selected:
                    selected.append(each.attribute)
        elif group_type == self.cfg.GROUPTYPES[2]:
            for each in group:
                if each.rect.collidepoint(mouse_pos):
                    each.is_selected = True
                    selected.append(each.attribute)
        else:
            raise ValueError('checkClicked.group_type unsupport %s, expect %s, %s or %s...' % (group_type, *self.cfg.GROUPTYPES))
        return selected
    '获取数字精灵组'

    def getNumberSpritesGroup(self, numbers):
        if False:
            while True:
                i = 10
        number_sprites_group = pygame.sprite.Group()
        for (idx, number) in enumerate(numbers):
            args = (*self.cfg.NUMBERCARD_POSITIONS[idx], str(number), self.resource_loader.fonts['number'], self.cfg.NUMBERFONT_COLORS, self.cfg.NUMBERCARD_COLORS, str(number))
            number_sprites_group.add(Card(*args))
        return number_sprites_group
    '获取运算符精灵组'

    def getOperatorSpritesGroup(self, operators):
        if False:
            return 10
        operator_sprites_group = pygame.sprite.Group()
        for (idx, operator) in enumerate(operators):
            args = (*self.cfg.OPERATORCARD_POSITIONS[idx], str(operator), self.resource_loader.fonts['default'], self.cfg.OPREATORFONT_COLORS, self.cfg.OPERATORCARD_COLORS, str(operator))
            operator_sprites_group.add(Card(*args))
        return operator_sprites_group
    '获取按钮精灵组'

    def getButtonSpritesGroup(self, buttons):
        if False:
            for i in range(10):
                print('nop')
        button_sprites_group = pygame.sprite.Group()
        for (idx, button) in enumerate(buttons):
            args = (*self.cfg.BUTTONCARD_POSITIONS[idx], str(button), self.resource_loader.fonts['default'], self.cfg.BUTTONFONT_COLORS, self.cfg.BUTTONCARD_COLORS, str(button), self.resource_loader.fonts['answer'])
            button_sprites_group.add(Button(*args))
        return button_sprites_group
    '计算'

    def calculate(self, number1, number2, operator):
        if False:
            i = 10
            return i + 15
        operator_map = {'+': '+', '-': '-', '×': '*', '÷': '/'}
        try:
            result = str(eval(number1 + operator_map[operator] + number2))
            return result if '.' not in result else str(Fraction(number1 + operator_map[operator] + number2))
        except:
            return None
    '在屏幕上显示信息'

    def showInfo(self, text, screen):
        if False:
            while True:
                i = 10
        rect = pygame.Rect(200, 180, 400, 200)
        pygame.draw.rect(screen, self.cfg.PAPAYAWHIP, rect)
        font = self.resource_loader.fonts['info']
        text_render = font.render(text, True, self.cfg.BLACK)
        font_size = font.size(text)
        screen.blit(text_render, (rect.x + (rect.width - font_size[0]) / 2, rect.y + (rect.height - font_size[1]) / 2))