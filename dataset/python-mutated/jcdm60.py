class PythagoreanTriplesFinder:

    def __init__(self, maximum):
        if False:
            i = 10
            return i + 15
        self.maximum = maximum
        self.triples = []

    def find_pythagorean_triples(self):
        if False:
            for i in range(10):
                print('nop')
        for a in range(1, self.maximum + 1):
            for b in range(a, self.maximum + 1):
                c_square = a ** 2 + b ** 2
                c = int(c_square ** 0.5)
                if c <= self.maximum and c_square == c ** 2:
                    self.triples.append((a, b, c))

    def get_triples(self):
        if False:
            i = 10
            return i + 15
        return self.triples
if __name__ == '__main__':
    max_value = 10
    triples_finder = PythagoreanTriplesFinder(max_value)
    triples_finder.find_pythagorean_triples()
    triples = triples_finder.get_triples()
    print(triples)