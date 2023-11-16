class World:

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.history = []

    def add_to_history(self, something):
        if False:
            i = 10
            return i + 15
        self.history.append(something)
world = World()
assert world.history == []
world.add_to_history('something')
assert world.history == ['something']