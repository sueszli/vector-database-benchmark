'''
Function:
    关卡切换界面
Author:
    Charles
微信公众号:
    Charles的皮卡丘
'''
import pygame
from .....utils import QuitGame


'''关卡切换界面'''
def SwitchLevelIterface(screen, cfg, resource_loader, level_next=1):
    background_img = resource_loader.images['others']['background']
    color_white = (255, 255, 255)
    color_gray = (192, 192, 192)
    font = resource_loader.fonts['switch']
    logo_img = resource_loader.images['others']['logo']
    logo_img = pygame.transform.scale(logo_img, (446, 70))
    logo_rect = logo_img.get_rect()
    logo_rect.centerx, logo_rect.centery = cfg.SCREENSIZE[0] / 2, cfg.SCREENSIZE[1] // 4
    # 游戏加载提示
    font_render = font.render('Loading game data, You will enter Level-%s' % level_next, True, color_white)
    font_rect = font_render.get_rect()
    font_rect.centerx, font_rect.centery = cfg.SCREENSIZE[0] / 2, cfg.SCREENSIZE[1] / 2
    # 游戏加载进度条
    gamebar = resource_loader.images['others']['gamebar'].convert_alpha()
    gamebar_rect = gamebar.get_rect()
    gamebar_rect.centerx, gamebar_rect.centery = cfg.SCREENSIZE[0] / 2, cfg.SCREENSIZE[1] / 1.4
    tank_cursor = resource_loader.images['player']['player1'][0].convert_alpha().subsurface((0, 144), (48, 48))
    tank_rect = tank_cursor.get_rect()
    tank_rect.left = gamebar_rect.left
    tank_rect.centery = gamebar_rect.centery
    # 加载所需时间
    load_time_left = gamebar_rect.right - tank_rect.right + 8
    # 主循环
    clock = pygame.time.Clock()
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                QuitGame()
        if load_time_left <= 0:
            return
        screen.blit(background_img, (0, 0))
        screen.blit(logo_img, logo_rect)
        screen.blit(font_render, font_rect)
        screen.blit(gamebar, gamebar_rect)
        screen.blit(tank_cursor, tank_rect)
        pygame.draw.rect(screen, color_gray, (gamebar_rect.left+8, gamebar_rect.top+8, tank_rect.left-gamebar_rect.left-8, tank_rect.bottom-gamebar_rect.top-16))
        tank_rect.left += 1
        load_time_left -= 1
        pygame.display.update()
        clock.tick(cfg.FPS)