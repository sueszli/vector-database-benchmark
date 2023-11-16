from turtle import Screen, Turtle, mainloop

class ColorTurtle(Turtle):

    def __init__(self, x, y):
        if False:
            print('Hello World!')
        Turtle.__init__(self)
        self.shape('turtle')
        self.resizemode('user')
        self.shapesize(3, 3, 5)
        self.pensize(10)
        self._color = [0, 0, 0]
        self.x = x
        self._color[x] = y
        self.color(self._color)
        self.speed(0)
        self.left(90)
        self.pu()
        self.goto(x, 0)
        self.pd()
        self.sety(1)
        self.pu()
        self.sety(y)
        self.pencolor('gray25')
        self.ondrag(self.shift)

    def shift(self, x, y):
        if False:
            return 10
        self.sety(max(0, min(y, 1)))
        self._color[self.x] = self.ycor()
        self.fillcolor(self._color)
        setbgcolor()

def setbgcolor():
    if False:
        while True:
            i = 10
    screen.bgcolor(red.ycor(), green.ycor(), blue.ycor())

def main():
    if False:
        while True:
            i = 10
    global screen, red, green, blue
    screen = Screen()
    screen.delay(0)
    screen.setworldcoordinates(-1, -0.3, 3, 1.3)
    red = ColorTurtle(0, 0.5)
    green = ColorTurtle(1, 0.5)
    blue = ColorTurtle(2, 0.5)
    setbgcolor()
    writer = Turtle()
    writer.ht()
    writer.pu()
    writer.goto(1, 1.15)
    writer.write('DRAG!', align='center', font=('Arial', 30, ('bold', 'italic')))
    return 'EVENTLOOP'
if __name__ == '__main__':
    msg = main()
    print(msg)
    mainloop()