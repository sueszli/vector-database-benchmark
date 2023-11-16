class HashableDict(dict):

    def __hash__(self):
        if False:
            i = 10
            return i + 15
        return hash(tuple(sorted(self.items())))