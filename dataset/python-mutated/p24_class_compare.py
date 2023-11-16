"""
Topic: 类支持比较操作
Desc : 
"""
from functools import total_ordering

class Room:

    def __init__(self, name, length, width):
        if False:
            return 10
        self.name = name
        self.length = length
        self.width = width
        self.square_feet = self.length * self.width

@total_ordering
class House:

    def __init__(self, name, style):
        if False:
            i = 10
            return i + 15
        self.name = name
        self.style = style
        self.rooms = list()

    @property
    def living_space_footage(self):
        if False:
            return 10
        return sum((r.square_feet for r in self.rooms))

    def add_room(self, room):
        if False:
            while True:
                i = 10
        self.rooms.append(room)

    def __str__(self):
        if False:
            i = 10
            return i + 15
        return '{}: {} square foot {}'.format(self.name, self.living_space_footage, self.style)

    def __eq__(self, other):
        if False:
            i = 10
            return i + 15
        return self.living_space_footage == other.living_space_footage

    def __lt__(self, other):
        if False:
            i = 10
            return i + 15
        return self.living_space_footage < other.living_space_footage
h1 = House('h1', 'Cape')
h1.add_room(Room('Master Bedroom', 14, 21))
h1.add_room(Room('Living Room', 18, 20))
h1.add_room(Room('Kitchen', 12, 16))
h1.add_room(Room('Office', 12, 12))
h2 = House('h2', 'Ranch')
h2.add_room(Room('Master Bedroom', 14, 21))
h2.add_room(Room('Living Room', 18, 20))
h2.add_room(Room('Kitchen', 12, 16))
h3 = House('h3', 'Split')
h3.add_room(Room('Master Bedroom', 14, 21))
h3.add_room(Room('Living Room', 18, 20))
h3.add_room(Room('Office', 12, 16))
h3.add_room(Room('Kitchen', 15, 17))
houses = [h1, h2, h3]
print('Is h1 bigger than h2?', h1 > h2)
print('Is h2 smaller than h3?', h2 < h3)
print('Is h2 greater than or equal to h1?', h2 >= h1)
print('Which one is biggest?', max(houses))
print('Which is smallest?', min(houses))