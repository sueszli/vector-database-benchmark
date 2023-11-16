a = 3
a
b = 3
b
c = 3
c
d = 'It should not read comments from the next line'
d
e = 'It should not read comments from the previous line'
e

class BB:
    pass

def test(a, b):
    if False:
        i = 10
        return i + 15
    a = a
    c = a
    d = a
    e = a
    a
    c
    d
    e

class AA:

    class BB:
        pass

def test(a):
    if False:
        return 10
    a

def test(a):
    if False:
        return 10
    a
(a, b) = (1, 2)
a
b

class Employee:
    pass
from typing import List, Tuple
x = []
x[1]
(x, y, z) = ([], [], [])
y[2]
(x, y, z) = ([], [], [])
for zi in z:
    zi
x = [1, 2]
x[1]
for bar in foo():
    bar
for (bar, baz) in foo():
    bar
    baz
for (bar, baz) in foo():
    ' type hinting on next line should not work '
    bar
    baz
with foo():
    ...
with foo() as f:
    f
with foo() as f:
    ' type hinting on next line should not work '
    f
aaa = some_extremely_long_function_name_that_doesnt_leave_room_for_hints()
aaa

class Dog:

    def __init__(self, age, friends, name):
        if False:
            return 10
        self.age = age
        self.friends = friends
        friends[0][1]
        self.name = name

    def friend_for_name(self, name):
        if False:
            i = 10
            return i + 15
        for (friend_name, friend) in self.friends:
            if friend_name == name:
                return friend
        raise ValueError()

    def bark(self):
        if False:
            while True:
                i = 10
        pass
buddy = Dog(UNKNOWN_NAME1, UNKNOWN_NAME2, UNKNOWN_NAME3)
friend = buddy.friend_for_name('buster')
friend.bark()
friend = buddy.friends[0][1]
friend.bark()
friend.name

def annot():
    if False:
        i = 10
        return i + 15
    pass
annot()
x = UNKNOWN_NAME2
x

class Cat(object):

    def __init__(self, age, friends, name):
        if False:
            while True:
                i = 10
        self.age = age
        self.friends = friends
        self.name = name
cat = Cat(UNKNOWN_NAME4, UNKNOWN_NAME5, UNKNOWN_NAME6)
cat.name

def x(a, b):
    if False:
        i = 10
        return i + 15
    a

def x(a, b):
    if False:
        for i in range(10):
            print('nop')
    a

def x(a, b, c):
    if False:
        i = 10
        return i + 15
    b
    c