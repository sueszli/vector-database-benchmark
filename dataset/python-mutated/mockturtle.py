"""Mock turtle module.
"""
state = {}
events = []

class Turtle:

    def __init__(self, visible=True):
        if False:
            return 10
        pass

    def goto(self, x, y):
        if False:
            while True:
                i = 10
        pass

    def up(self):
        if False:
            i = 10
            return i + 15
        pass

    def down(self):
        if False:
            i = 10
            return i + 15
        pass

    def width(self, size):
        if False:
            for i in range(10):
                print('nop')
        pass

    def color(self, fill, outline=None):
        if False:
            return 10
        pass

    def write(self, text, font=None, align=None):
        if False:
            for i in range(10):
                print('nop')
        pass

    def begin_fill(self):
        if False:
            print('Hello World!')
        pass

    def end_fill(self):
        if False:
            return 10
        pass

    def forward(self, steps):
        if False:
            while True:
                i = 10
        pass

    def back(self, steps):
        if False:
            while True:
                i = 10
        pass

    def addshape(self, reference):
        if False:
            while True:
                i = 10
        pass

    def shape(self, reference):
        if False:
            for i in range(10):
                print('nop')
        pass

    def stamp(self):
        if False:
            while True:
                i = 10
        pass

    def left(self, degrees):
        if False:
            print('Hello World!')
        pass

    def right(self, degrees):
        if False:
            i = 10
            return i + 15
        pass

    def hideturtle(self):
        if False:
            while True:
                i = 10
        pass

    def tracer(self, state):
        if False:
            while True:
                i = 10
        pass

    def dot(self, size, color=None):
        if False:
            while True:
                i = 10
        pass

    def clear(self):
        if False:
            print('Hello World!')
        pass

    def circle(self, radius):
        if False:
            while True:
                i = 10
        pass

    def bgcolor(self, name):
        if False:
            for i in range(10):
                print('nop')
        pass

    def update(self):
        if False:
            return 10
        pass

    def undo(self):
        if False:
            for i in range(10):
                print('nop')
        pass
_turtle = Turtle()
goto = _turtle.goto
up = _turtle.up
down = _turtle.down
width = _turtle.width
color = _turtle.color
write = _turtle.write
begin_fill = _turtle.begin_fill
end_fill = _turtle.end_fill
forward = _turtle.forward
back = _turtle.back
addshape = _turtle.addshape
shape = _turtle.shape
stamp = _turtle.stamp
left = _turtle.left
right = _turtle.right
hideturtle = _turtle.hideturtle
tracer = _turtle.tracer
dot = _turtle.dot
clear = _turtle.clear
circle = _turtle.circle
bgcolor = _turtle.bgcolor
update = _turtle.update
undo = _turtle.undo

def setup(width, height, x, y):
    if False:
        print('Hello World!')
    pass

def listen():
    if False:
        while True:
            i = 10
    pass

def onkey(function, key):
    if False:
        while True:
            i = 10
    state['key ' + key] = function

def ontimer(function, delay):
    if False:
        while True:
            i = 10
    state['timer'] = function
    state['delay'] = delay

def onscreenclick(function):
    if False:
        return 10
    state['click'] = function

def done():
    if False:
        i = 10
        return i + 15
    for event in events:
        name = event[0]
        try:
            function = state[name]
        except KeyError:
            if name == 'timer':
                if event[1:] and event[1]:
                    continue
            raise
        if name == 'timer':
            del state['timer']
            del state['delay']
            args = ()
        else:
            args = event[1:]
        function(*args)