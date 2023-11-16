import pyxel

class App:

    def __init__(self):
        if False:
            while True:
                i = 10
        pyxel.init(160, 120, title='Hello Pyxel')
        pyxel.image(0).load(0, 0, 'assets/pyxel_logo_38x16.png')
        pyxel.run(self.update, self.draw)

    def update(self):
        if False:
            i = 10
            return i + 15
        if pyxel.btnp(pyxel.KEY_Q):
            pyxel.quit()

    def draw(self):
        if False:
            i = 10
            return i + 15
        pyxel.cls(0)
        pyxel.text(55, 41, 'Hello, Pyxel!', pyxel.frame_count % 16)
        pyxel.blt(61, 66, 0, 0, 0, 38, 16)
App()