"""
Function:
    定义皮卡丘类
Author:
    Charles
微信公众号:
    Charles的皮卡丘
"""
import cocos
'皮卡丘类'

class Pikachu(cocos.sprite.Sprite):

    def __init__(self, imagepath, **kwargs):
        if False:
            print('Hello World!')
        super(Pikachu, self).__init__(imagepath)
        self.image_anchor = (0, 0)
        self.reset(False)
        self.schedule(self.update)
    '声控跳跃'

    def jump(self, h):
        if False:
            i = 10
            return i + 15
        if self.is_able_jump:
            self.y += 1
            self.speed -= max(min(h, 10), 7)
            self.is_able_jump = False
    '着陆后静止'

    def land(self, y):
        if False:
            while True:
                i = 10
        if self.y > y - 25:
            self.is_able_jump = True
            self.speed = 0
            self.y = y
    '更新(重力下降)'

    def update(self, dt):
        if False:
            return 10
        self.speed += 10 * dt
        self.y -= self.speed
        if self.y < -85:
            self.reset()
    '重置'

    def reset(self, flag=True):
        if False:
            return 10
        if flag:
            self.parent.reset()
        self.is_able_jump = False
        self.speed = 0
        self.position = (80, 280)