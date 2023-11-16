"""Test out HMMs using the Occasionally Dishonest Casino.

This uses the occasionally dishonest casino example from Biological
Sequence Analysis by Durbin et al.

In this example, we are dealing with a casino that has two types of
dice, a fair dice that has 1/6 probability of rolling any number and
a loaded dice that has 1/2 probability to roll a 6, and 1/10 probability
to roll any other number. The probability of switching from the fair to
loaded dice is .05 and the probability of switching from loaded to fair is
.1.
"""
import random
import unittest
from Bio.HMM import MarkovModel
from Bio.HMM import Trainer
from Bio.HMM import Utilities
VERBOSE = 0
dice_roll_alphabet = ('1', '2', '3', '4', '5', '6')
dice_type_alphabet = ('F', 'L')

def generate_rolls(num_rolls):
    if False:
        i = 10
        return i + 15
    'Generate a bunch of rolls corresponding to the casino probabilities.\n\n    Returns:\n    - The generate roll sequence\n    - The state sequence that generated the roll.\n\n    '
    cur_state = 'F'
    roll_seq = []
    state_seq = []
    loaded_weights = [0.1, 0.1, 0.1, 0.1, 0.1, 0.5]
    for roll in range(num_rolls):
        state_seq.append(cur_state)
        if cur_state == 'F':
            new_rolls = random.choices(dice_roll_alphabet)
        elif cur_state == 'L':
            new_rolls = random.choices(dice_roll_alphabet, weights=loaded_weights)
        new_roll = new_rolls[0]
        roll_seq.append(new_roll)
        chance_num = random.random()
        if cur_state == 'F':
            if chance_num <= 0.05:
                cur_state = 'L'
        elif cur_state == 'L':
            if chance_num <= 0.1:
                cur_state = 'F'
    return (roll_seq, state_seq)

class TestHMMCasino(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if False:
            print('Hello World!')
        cls.mm_builder = MarkovModel.MarkovModelBuilder(dice_type_alphabet, dice_roll_alphabet)
        cls.mm_builder.allow_all_transitions()
        cls.mm_builder.set_random_probabilities()
        (cls.rolls, cls.states) = generate_rolls(3000)

    def test_baum_welch_training_standard(self):
        if False:
            i = 10
            return i + 15
        'Standard Training with known states.'
        known_training_seq = Trainer.TrainingSequence(self.rolls, self.states)
        standard_mm = self.mm_builder.get_markov_model()
        trainer = Trainer.KnownStateTrainer(standard_mm)
        trained_mm = trainer.train([known_training_seq])
        if VERBOSE:
            print(trained_mm.transition_prob)
            print(trained_mm.emission_prob)
        (test_rolls, test_states) = generate_rolls(300)
        (predicted_states, prob) = trained_mm.viterbi(test_rolls, dice_type_alphabet)
        if VERBOSE:
            print(f'Prediction probability: {prob:f}')
            Utilities.pretty_print_prediction(test_rolls, test_states, predicted_states)

    def test_baum_welch_training_without(self):
        if False:
            i = 10
            return i + 15
        'Baum-Welch training without known state sequences.'
        training_seq = Trainer.TrainingSequence(self.rolls, ())

        def stop_training(log_likelihood_change, num_iterations):
            if False:
                i = 10
                return i + 15
            'Tell the training model when to stop.'
            if VERBOSE:
                print(f'll change: {log_likelihood_change:f}')
            if log_likelihood_change < 0.01:
                return 1
            elif num_iterations >= 10:
                return 1
            else:
                return 0
        baum_welch_mm = self.mm_builder.get_markov_model()
        trainer = Trainer.BaumWelchTrainer(baum_welch_mm)
        trained_mm = trainer.train([training_seq], stop_training)
        if VERBOSE:
            print(trained_mm.transition_prob)
            print(trained_mm.emission_prob)
        (test_rolls, test_states) = generate_rolls(300)
        (predicted_states, prob) = trained_mm.viterbi(test_rolls, dice_type_alphabet)
        if VERBOSE:
            print(f'Prediction probability: {prob:f}')
            Utilities.pretty_print_prediction(self.test_rolls, test_states, predicted_states)
if __name__ == '__main__':
    runner = unittest.TextTestRunner(verbosity=2)
    unittest.main(testRunner=runner)