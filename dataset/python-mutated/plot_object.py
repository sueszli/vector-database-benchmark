class PlotObject:
    """
    Base class for objects which can be displayed in
    a Plot.
    """
    visible = True

    def _draw(self):
        if False:
            while True:
                i = 10
        if self.visible:
            self.draw()

    def draw(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        OpenGL rendering code for the plot object.\n        Override in base class.\n        '
        pass