""" pygame.examples.aliens

Shows a mini game where you have to defend against aliens.

What does it show you about pygame?

* pg.sprite, the difference between Sprite and Group.
* dirty rectangle optimization for processing for speed.
* music with pg.mixer.music, including fadeout
* sound effects with pg.Sound
* event processing, keyboard handling, QUIT handling.
* a main loop frame limited with a game clock from pg.time.Clock
* fullscreen switching.


Controls
--------

* Left and right arrows to move.
* Space bar to shoot
* f key to toggle between fullscreen.

"""
import os
import random
from typing import List
import pygame as pg
if not pg.image.get_extended():
    raise SystemExit('Sorry, extended image module required')
MAX_SHOTS = 2
ALIEN_ODDS = 22
BOMB_ODDS = 60
ALIEN_RELOAD = 12
SCREENRECT = pg.Rect(0, 0, 640, 480)
SCORE = 0
main_dir = os.path.split(os.path.abspath(__file__))[0]

def load_image(file):
    if False:
        return 10
    'loads an image, prepares it for play'
    file = os.path.join(main_dir, 'data', file)
    try:
        surface = pg.image.load(file)
    except pg.error:
        raise SystemExit(f'Could not load image "{file}" {pg.get_error()}')
    return surface.convert()

def load_sound(file):
    if False:
        while True:
            i = 10
    'because pygame can be compiled without mixer.'
    if not pg.mixer:
        return None
    file = os.path.join(main_dir, 'data', file)
    try:
        sound = pg.mixer.Sound(file)
        return sound
    except pg.error:
        print(f'Warning, unable to load, {file}')
    return None

class Player(pg.sprite.Sprite):
    """Representing the player as a moon buggy type car."""
    speed = 10
    bounce = 24
    gun_offset = -11
    images: List[pg.Surface] = []

    def __init__(self, *groups):
        if False:
            i = 10
            return i + 15
        pg.sprite.Sprite.__init__(self, *groups)
        self.image = self.images[0]
        self.rect = self.image.get_rect(midbottom=SCREENRECT.midbottom)
        self.reloading = 0
        self.origtop = self.rect.top
        self.facing = -1

    def move(self, direction):
        if False:
            while True:
                i = 10
        if direction:
            self.facing = direction
        self.rect.move_ip(direction * self.speed, 0)
        self.rect = self.rect.clamp(SCREENRECT)
        if direction < 0:
            self.image = self.images[0]
        elif direction > 0:
            self.image = self.images[1]
        self.rect.top = self.origtop - self.rect.left // self.bounce % 2

    def gunpos(self):
        if False:
            return 10
        pos = self.facing * self.gun_offset + self.rect.centerx
        return (pos, self.rect.top)

class Alien(pg.sprite.Sprite):
    """An alien space ship. That slowly moves down the screen."""
    speed = 13
    animcycle = 12
    images: List[pg.Surface] = []

    def __init__(self, *groups):
        if False:
            i = 10
            return i + 15
        pg.sprite.Sprite.__init__(self, *groups)
        self.image = self.images[0]
        self.rect = self.image.get_rect()
        self.facing = random.choice((-1, 1)) * Alien.speed
        self.frame = 0
        if self.facing < 0:
            self.rect.right = SCREENRECT.right

    def update(self):
        if False:
            i = 10
            return i + 15
        self.rect.move_ip(self.facing, 0)
        if not SCREENRECT.contains(self.rect):
            self.facing = -self.facing
            self.rect.top = self.rect.bottom + 1
            self.rect = self.rect.clamp(SCREENRECT)
        self.frame = self.frame + 1
        self.image = self.images[self.frame // self.animcycle % 3]

class Explosion(pg.sprite.Sprite):
    """An explosion. Hopefully the Alien and not the player!"""
    defaultlife = 12
    animcycle = 3
    images: List[pg.Surface] = []

    def __init__(self, actor, *groups):
        if False:
            i = 10
            return i + 15
        pg.sprite.Sprite.__init__(self, *groups)
        self.image = self.images[0]
        self.rect = self.image.get_rect(center=actor.rect.center)
        self.life = self.defaultlife

    def update(self):
        if False:
            while True:
                i = 10
        "called every time around the game loop.\n\n        Show the explosion surface for 'defaultlife'.\n        Every game tick(update), we decrease the 'life'.\n\n        Also we animate the explosion.\n        "
        self.life = self.life - 1
        self.image = self.images[self.life // self.animcycle % 2]
        if self.life <= 0:
            self.kill()

class Shot(pg.sprite.Sprite):
    """a bullet the Player sprite fires."""
    speed = -11
    images: List[pg.Surface] = []

    def __init__(self, pos, *groups):
        if False:
            return 10
        pg.sprite.Sprite.__init__(self, *groups)
        self.image = self.images[0]
        self.rect = self.image.get_rect(midbottom=pos)

    def update(self):
        if False:
            print('Hello World!')
        'called every time around the game loop.\n\n        Every tick we move the shot upwards.\n        '
        self.rect.move_ip(0, self.speed)
        if self.rect.top <= 0:
            self.kill()

class Bomb(pg.sprite.Sprite):
    """A bomb the aliens drop."""
    speed = 9
    images: List[pg.Surface] = []

    def __init__(self, alien, explosion_group, *groups):
        if False:
            for i in range(10):
                print('nop')
        pg.sprite.Sprite.__init__(self, *groups)
        self.image = self.images[0]
        self.rect = self.image.get_rect(midbottom=alien.rect.move(0, 5).midbottom)
        self.explosion_group = explosion_group

    def update(self):
        if False:
            print('Hello World!')
        "called every time around the game loop.\n\n        Every frame we move the sprite 'rect' down.\n        When it reaches the bottom we:\n\n        - make an explosion.\n        - remove the Bomb.\n        "
        self.rect.move_ip(0, self.speed)
        if self.rect.bottom >= 470:
            Explosion(self, self.explosion_group)
            self.kill()

class Score(pg.sprite.Sprite):
    """to keep track of the score."""

    def __init__(self, *groups):
        if False:
            while True:
                i = 10
        pg.sprite.Sprite.__init__(self, *groups)
        self.font = pg.font.Font(None, 20)
        self.font.set_italic(1)
        self.color = 'white'
        self.lastscore = -1
        self.update()
        self.rect = self.image.get_rect().move(10, 450)

    def update(self):
        if False:
            print('Hello World!')
        'We only update the score in update() when it has changed.'
        if SCORE != self.lastscore:
            self.lastscore = SCORE
            msg = f'Score: {SCORE}'
            self.image = self.font.render(msg, 0, self.color)

def main(winstyle=0):
    if False:
        return 10
    if pg.get_sdl_version()[0] == 2:
        pg.mixer.pre_init(44100, 32, 2, 1024)
    pg.init()
    if pg.mixer and (not pg.mixer.get_init()):
        print('Warning, no sound')
        pg.mixer = None
    fullscreen = False
    winstyle = 0
    bestdepth = pg.display.mode_ok(SCREENRECT.size, winstyle, 32)
    screen = pg.display.set_mode(SCREENRECT.size, winstyle, bestdepth)
    img = load_image('player1.gif')
    Player.images = [img, pg.transform.flip(img, 1, 0)]
    img = load_image('explosion1.gif')
    Explosion.images = [img, pg.transform.flip(img, 1, 1)]
    Alien.images = [load_image(im) for im in ('alien1.gif', 'alien2.gif', 'alien3.gif')]
    Bomb.images = [load_image('bomb.gif')]
    Shot.images = [load_image('shot.gif')]
    icon = pg.transform.scale(Alien.images[0], (32, 32))
    pg.display.set_icon(icon)
    pg.display.set_caption('Pygame Aliens')
    pg.mouse.set_visible(0)
    bgdtile = load_image('background.gif')
    background = pg.Surface(SCREENRECT.size)
    for x in range(0, SCREENRECT.width, bgdtile.get_width()):
        background.blit(bgdtile, (x, 0))
    screen.blit(background, (0, 0))
    pg.display.flip()
    boom_sound = load_sound('boom.wav')
    shoot_sound = load_sound('car_door.wav')
    if pg.mixer:
        music = os.path.join(main_dir, 'data', 'house_lo.wav')
        pg.mixer.music.load(music)
        pg.mixer.music.play(-1)
    aliens = pg.sprite.Group()
    shots = pg.sprite.Group()
    bombs = pg.sprite.Group()
    all = pg.sprite.RenderUpdates()
    lastalien = pg.sprite.GroupSingle()
    alienreload = ALIEN_RELOAD
    clock = pg.time.Clock()
    global SCORE
    player = Player(all)
    Alien(aliens, all, lastalien)
    if pg.font:
        all.add(Score(all))
    while player.alive():
        for event in pg.event.get():
            if event.type == pg.QUIT:
                return
            if event.type == pg.KEYDOWN and event.key == pg.K_ESCAPE:
                return
            if event.type == pg.KEYDOWN:
                if event.key == pg.K_f:
                    if not fullscreen:
                        print('Changing to FULLSCREEN')
                        screen_backup = screen.copy()
                        screen = pg.display.set_mode(SCREENRECT.size, winstyle | pg.FULLSCREEN, bestdepth)
                        screen.blit(screen_backup, (0, 0))
                    else:
                        print('Changing to windowed mode')
                        screen_backup = screen.copy()
                        screen = pg.display.set_mode(SCREENRECT.size, winstyle, bestdepth)
                        screen.blit(screen_backup, (0, 0))
                    pg.display.flip()
                    fullscreen = not fullscreen
        keystate = pg.key.get_pressed()
        all.clear(screen, background)
        all.update()
        direction = keystate[pg.K_RIGHT] - keystate[pg.K_LEFT]
        player.move(direction)
        firing = keystate[pg.K_SPACE]
        if not player.reloading and firing and (len(shots) < MAX_SHOTS):
            Shot(player.gunpos(), shots, all)
            if pg.mixer and shoot_sound is not None:
                shoot_sound.play()
        player.reloading = firing
        if alienreload:
            alienreload = alienreload - 1
        elif not int(random.random() * ALIEN_ODDS):
            Alien(aliens, all, lastalien)
            alienreload = ALIEN_RELOAD
        if lastalien and (not int(random.random() * BOMB_ODDS)):
            Bomb(lastalien.sprite, all, bombs, all)
        for alien in pg.sprite.spritecollide(player, aliens, 1):
            if pg.mixer and boom_sound is not None:
                boom_sound.play()
            Explosion(alien, all)
            Explosion(player, all)
            SCORE = SCORE + 1
            player.kill()
        for alien in pg.sprite.groupcollide(aliens, shots, 1, 1).keys():
            if pg.mixer and boom_sound is not None:
                boom_sound.play()
            Explosion(alien, all)
            SCORE = SCORE + 1
        for bomb in pg.sprite.spritecollide(player, bombs, 1):
            if pg.mixer and boom_sound is not None:
                boom_sound.play()
            Explosion(player, all)
            Explosion(bomb, all)
            player.kill()
        dirty = all.draw(screen)
        pg.display.update(dirty)
        clock.tick(40)
    if pg.mixer:
        pg.mixer.music.fadeout(1000)
    pg.time.wait(1000)
if __name__ == '__main__':
    main()
    pg.quit()