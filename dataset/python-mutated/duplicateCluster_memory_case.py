import random
import dedupe.core
import dedupe.dedupe

def candidates_gen():
    if False:
        print('Hello World!')
    candidate_set = set([])
    for _ in range(10 ** 5):
        block = [((random.randint(0, 1000), 'a'), (random.randint(0, 1000), 'b'))]
        for candidate in block:
            pair_ids = (candidate[0][0], candidate[1][0])
            if pair_ids not in candidate_set:
                yield candidate
                candidate_set.add(pair_ids)
    del candidate_set

@profile
def generator_test():
    if False:
        for i in range(10):
            print('nop')
    a = sum((candidate[0][0] for candidate in candidates_gen()))
    print(a)
generator_test()