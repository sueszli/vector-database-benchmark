""" pygame.examples.chimp

This simple example is used for the line-by-line tutorial
that comes with pygame. It is based on a 'popular' web banner.
Note there are comments here, but for the full explanation,
follow along in the tutorial.
"""
import os
import pygame as pg
if not pg.font:
    print('Warning, fonts disabled')
if not pg.mixer:
    print('Warning, sound disabled')
main_dir = os.path.split(os.path.abspath(__file__))[0]
data_dir = os.path.join(main_dir, 'data')

def load_image(name, colorkey=None, scale=1):
    if False:
        return 10
    fullname = os.path.join(data_dir, name)
    image = pg.image.load(fullname)
    image = image.convert()
    size = image.get_size()
    size = (size[0] * scale, size[1] * scale)
    image = pg.transform.scale(image, size)
    if colorkey is not None:
        if colorkey == -1:
            colorkey = image.get_at((0, 0))
        image.set_colorkey(colorkey, pg.RLEACCEL)
    return (image, image.get_rect())

def load_sound(name):
    if False:
        while True:
            i = 10

    class NoneSound:

        def play(self):
            if False:
                return 10
            pass
    if not pg.mixer or not pg.mixer.get_init():
        return NoneSound()
    fullname = os.path.join(data_dir, name)
    sound = pg.mixer.Sound(fullname)
    return sound

class Fist(pg.sprite.Sprite):
    """moves a clenched fist on the screen, following the mouse"""

    def __init__(self):
        if False:
            print('Hello World!')
        pg.sprite.Sprite.__init__(self)
        (self.image, self.rect) = load_image('fist.png', -1)
        self.fist_offset = (-235, -80)
        self.punching = False

    def update(self):
        if False:
            while True:
                i = 10
        'move the fist based on the mouse position'
        pos = pg.mouse.get_pos()
        self.rect.topleft = pos
        self.rect.move_ip(self.fist_offset)
        if self.punching:
            self.rect.move_ip(15, 25)

    def punch(self, target):
        if False:
            i = 10
            return i + 15
        'returns true if the fist collides with the target'
        if not self.punching:
            self.punching = True
            hitbox = self.rect.inflate(-5, -5)
            return hitbox.colliderect(target.rect)

    def unpunch(self):
        if False:
            i = 10
            return i + 15
        'called to pull the fist back'
        self.punching = False

class Chimp(pg.sprite.Sprite):
    """moves a monkey critter across the screen. it can spin the
    monkey when it is punched."""

    def __init__(self):
        if False:
            return 10
        pg.sprite.Sprite.__init__(self)
        (self.image, self.rect) = load_image('chimp.png', -1, 4)
        screen = pg.display.get_surface()
        self.area = screen.get_rect()
        self.rect.topleft = (10, 90)
        self.move = 18
        self.dizzy = False

    def update(self):
        if False:
            for i in range(10):
                print('nop')
        'walk or spin, depending on the monkeys state'
        if self.dizzy:
            self._spin()
        else:
            self._walk()

    def _walk(self):
        if False:
            for i in range(10):
                print('nop')
        'move the monkey across the screen, and turn at the ends'
        newpos = self.rect.move((self.move, 0))
        if not self.area.contains(newpos):
            if self.rect.left < self.area.left or self.rect.right > self.area.right:
                self.move = -self.move
                newpos = self.rect.move((self.move, 0))
                self.image = pg.transform.flip(self.image, True, False)
        self.rect = newpos

    def _spin(self):
        if False:
            print('Hello World!')
        'spin the monkey image'
        center = self.rect.center
        self.dizzy = self.dizzy + 12
        if self.dizzy >= 360:
            self.dizzy = False
            self.image = self.original
        else:
            rotate = pg.transform.rotate
            self.image = rotate(self.original, self.dizzy)
        self.rect = self.image.get_rect(center=center)

    def punched(self):
        if False:
            return 10
        'this will cause the monkey to start spinning'
        if not self.dizzy:
            self.dizzy = True
            self.original = self.image

def main():
    if False:
        for i in range(10):
            print('nop')
    'this function is called when the program starts.\n    it initializes everything it needs, then runs in\n    a loop until the function returns.'
    pg.init()
    screen = pg.display.set_mode((1280, 480), pg.SCALED)
    pg.display.set_caption('Monkey Fever')
    pg.mouse.set_visible(False)
    background = pg.Surface(screen.get_size())
    background = background.convert()
    background.fill((170, 238, 187))
    if pg.font:
        font = pg.font.Font(None, 64)
        text = font.render('Pummel The Chimp, And Win $$$', True, (10, 10, 10))
        textpos = text.get_rect(centerx=background.get_width() / 2, y=10)
        background.blit(text, textpos)
    screen.blit(background, (0, 0))
    pg.display.flip()
    whiff_sound = load_sound('whiff.wav')
    punch_sound = load_sound('punch.wav')
    chimp = Chimp()
    fist = Fist()
    allsprites = pg.sprite.RenderPlain((chimp, fist))
    clock = pg.time.Clock()
    going = True
    while going:
        clock.tick(60)
        for event in pg.event.get():
            if event.type == pg.QUIT:
                going = False
            elif event.type == pg.KEYDOWN and event.key == pg.K_ESCAPE:
                going = False
            elif event.type == pg.MOUSEBUTTONDOWN:
                if fist.punch(chimp):
                    punch_sound.play()
                    chimp.punched()
                else:
                    whiff_sound.play()
            elif event.type == pg.MOUSEBUTTONUP:
                fist.unpunch()
        allsprites.update()
        screen.blit(background, (0, 0))
        allsprites.draw(screen)
        pg.display.flip()
    pg.quit()
if __name__ == '__main__':
    main()