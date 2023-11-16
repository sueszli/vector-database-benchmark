"""       turtle-example-suite:

         tdemo_minimal_hanoi.py

A minimal 'Towers of Hanoi' animation:
A tower of 6 discs is transferred from the
left to the right peg.

An imho quite elegant and concise
implementation using a tower class, which
is derived from the built-in type list.

Discs are turtles with shape "square", but
stretched to rectangles by shapesize()
 ---------------------------------------
       To exit press STOP button
 ---------------------------------------
"""
from turtle import *

class Disc(Turtle):

    def __init__(self, n):
        if False:
            i = 10
            return i + 15
        Turtle.__init__(self, shape='square', visible=False)
        self.pu()
        self.shapesize(1.5, n * 1.5, 2)
        self.fillcolor(n / 6.0, 0, 1 - n / 6.0)
        self.st()

class Tower(list):
    """Hanoi tower, a subclass of built-in type list"""

    def __init__(self, x):
        if False:
            while True:
                i = 10
        'create an empty tower. x is x-position of peg'
        self.x = x

    def push(self, d):
        if False:
            while True:
                i = 10
        d.setx(self.x)
        d.sety(-150 + 34 * len(self))
        self.append(d)

    def pop(self):
        if False:
            i = 10
            return i + 15
        d = list.pop(self)
        d.sety(150)
        return d

def hanoi(n, from_, with_, to_):
    if False:
        i = 10
        return i + 15
    if n > 0:
        hanoi(n - 1, from_, to_, with_)
        to_.push(from_.pop())
        hanoi(n - 1, with_, from_, to_)

def play():
    if False:
        while True:
            i = 10
    onkey(None, 'space')
    clear()
    try:
        hanoi(6, t1, t2, t3)
        write('press STOP button to exit', align='center', font=('Courier', 16, 'bold'))
    except Terminator:
        pass

def main():
    if False:
        i = 10
        return i + 15
    global t1, t2, t3
    ht()
    penup()
    goto(0, -225)
    t1 = Tower(-250)
    t2 = Tower(0)
    t3 = Tower(250)
    for i in range(6, 0, -1):
        t1.push(Disc(i))
    write('press spacebar to start game', align='center', font=('Courier', 16, 'bold'))
    onkey(play, 'space')
    listen()
    return 'EVENTLOOP'
if __name__ == '__main__':
    msg = main()
    print(msg)
    mainloop()