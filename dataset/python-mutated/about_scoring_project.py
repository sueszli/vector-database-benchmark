from runner.koan import *

def score(dice):
    if False:
        i = 10
        return i + 15
    pass

class AboutScoringProject(Koan):

    def test_score_of_an_empty_list_is_zero(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(0, score([]))

    def test_score_of_a_single_roll_of_5_is_50(self):
        if False:
            while True:
                i = 10
        self.assertEqual(50, score([5]))

    def test_score_of_a_single_roll_of_1_is_100(self):
        if False:
            while True:
                i = 10
        self.assertEqual(100, score([1]))

    def test_score_of_multiple_1s_and_5s_is_the_sum_of_individual_scores(self):
        if False:
            return 10
        self.assertEqual(300, score([1, 5, 5, 1]))

    def test_score_of_single_2s_3s_4s_and_6s_are_zero(self):
        if False:
            return 10
        self.assertEqual(0, score([2, 3, 4, 6]))

    def test_score_of_a_triple_1_is_1000(self):
        if False:
            print('Hello World!')
        self.assertEqual(1000, score([1, 1, 1]))

    def test_score_of_other_triples_is_100x(self):
        if False:
            while True:
                i = 10
        self.assertEqual(200, score([2, 2, 2]))
        self.assertEqual(300, score([3, 3, 3]))
        self.assertEqual(400, score([4, 4, 4]))
        self.assertEqual(500, score([5, 5, 5]))
        self.assertEqual(600, score([6, 6, 6]))

    def test_score_of_mixed_is_sum(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(250, score([2, 5, 2, 2, 3]))
        self.assertEqual(550, score([5, 5, 5, 5]))
        self.assertEqual(1150, score([1, 1, 1, 5, 1]))

    def test_ones_not_left_out(self):
        if False:
            print('Hello World!')
        self.assertEqual(300, score([1, 2, 2, 2]))
        self.assertEqual(350, score([1, 5, 2, 2, 2]))