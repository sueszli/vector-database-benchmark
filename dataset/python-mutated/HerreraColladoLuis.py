def is_triple(triple: list) -> bool:
    if False:
        print('Hello World!')
    if len(triple) == 3:
        return triple[0] * triple[0] + triple[1] * triple[1] == triple[2] * triple[2]

def find_triples(max: int) -> list:
    if False:
        print('Hello World!')
    numbers: list = []
    for i in range(1, max + 1):
        numbers.append(i)

    def find_triple(start: int, triple: list):
        if False:
            print('Hello World!')
        if is_triple(triple):
            triples.append(triple[:])
            return
        if start > max:
            return
        for index in range(start, len(numbers)):
            if index > start and numbers[index] == numbers[index - 1]:
                continue
            triple.append(numbers[index])
            find_triple(index + 1, triple)
            triple.pop()
    triples: list = []
    find_triple(0, [])
    return triples
print(find_triples(10))