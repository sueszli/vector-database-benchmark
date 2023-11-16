"""Dynamic Programming algorithms for general usage.

This module contains classes which implement Dynamic Programming
algorithms that can be used generally.
"""

class AbstractDPAlgorithms:
    """An abstract class to calculate forward and backward probabilities.

    This class should not be instantiated directly, but should be used
    through a derived class which implements proper scaling of variables.

    This class is just meant to encapsulate the basic forward and backward
    algorithms, and allow derived classes to deal with the problems of
    multiplying probabilities.

    Derived class of this must implement:

    - _forward_recursion -- Calculate the forward values in the recursion
      using some kind of technique for preventing underflow errors.
    - _backward_recursion -- Calculate the backward values in the recursion
      step using some technique to prevent underflow errors.

    """

    def __init__(self, markov_model, sequence):
        if False:
            while True:
                i = 10
        'Initialize to calculate forward and backward probabilities.\n\n        Arguments:\n         - markov_model -- The current Markov model we are working with.\n         - sequence -- A training sequence containing a set of emissions.\n\n        '
        self._mm = markov_model
        self._seq = sequence

    def _forward_recursion(self, cur_state, sequence_pos, forward_vars):
        if False:
            i = 10
            return i + 15
        'Calculate the forward recursion value (PRIVATE).'
        raise NotImplementedError('Subclasses must implement')

    def forward_algorithm(self):
        if False:
            print('Hello World!')
        'Calculate sequence probability using the forward algorithm.\n\n        This implements the forward algorithm, as described on p57-58 of\n        Durbin et al.\n\n        Returns:\n         - A dictionary containing the forward variables. This has keys of the\n           form (state letter, position in the training sequence), and values\n           containing the calculated forward variable.\n         - The calculated probability of the sequence.\n\n        '
        state_letters = self._mm.state_alphabet
        forward_var = {}
        forward_var[state_letters[0], -1] = 1
        for k in range(1, len(state_letters)):
            forward_var[state_letters[k], -1] = 0
        for i in range(len(self._seq.emissions)):
            for main_state in state_letters:
                forward_value = self._forward_recursion(main_state, i, forward_var)
                if forward_value is not None:
                    forward_var[main_state, i] = forward_value
        first_state = state_letters[0]
        seq_prob = 0
        for state_item in state_letters:
            forward_value = forward_var[state_item, len(self._seq.emissions) - 1]
            transition_value = self._mm.transition_prob[state_item, first_state]
            seq_prob += forward_value * transition_value
        return (forward_var, seq_prob)

    def _backward_recursion(self, cur_state, sequence_pos, forward_vars):
        if False:
            return 10
        'Calculate the backward recursion value (PRIVATE).'
        raise NotImplementedError('Subclasses must implement')

    def backward_algorithm(self):
        if False:
            for i in range(10):
                print('nop')
        'Calculate sequence probability using the backward algorithm.\n\n        This implements the backward algorithm, as described on p58-59 of\n        Durbin et al.\n\n        Returns:\n         - A dictionary containing the backwards variables. This has keys\n           of the form (state letter, position in the training sequence),\n           and values containing the calculated backward variable.\n\n        '
        state_letters = self._mm.state_alphabet
        backward_var = {}
        first_letter = state_letters[0]
        for state in state_letters:
            backward_var[state, len(self._seq.emissions) - 1] = self._mm.transition_prob[state, state_letters[0]]
        all_indexes = list(range(len(self._seq.emissions) - 1))
        all_indexes.reverse()
        for i in all_indexes:
            for main_state in state_letters:
                backward_value = self._backward_recursion(main_state, i, backward_var)
                if backward_value is not None:
                    backward_var[main_state, i] = backward_value
        return backward_var

class ScaledDPAlgorithms(AbstractDPAlgorithms):
    """Implement forward and backward algorithms using a rescaling approach.

    This scales the f and b variables, so that they remain within a
    manageable numerical interval during calculations. This approach is
    described in Durbin et al. on p 78.

    This approach is a little more straightforward then log transformation
    but may still give underflow errors for some types of models. In these
    cases, the LogDPAlgorithms class should be used.
    """

    def __init__(self, markov_model, sequence):
        if False:
            i = 10
            return i + 15
        'Initialize the scaled approach to calculating probabilities.\n\n        Arguments:\n         - markov_model -- The current Markov model we are working with.\n         - sequence -- A TrainingSequence object that must have a\n           set of emissions to work with.\n\n        '
        AbstractDPAlgorithms.__init__(self, markov_model, sequence)
        self._s_values = {}

    def _calculate_s_value(self, seq_pos, previous_vars):
        if False:
            i = 10
            return i + 15
        'Calculate the next scaling variable for a sequence position (PRIVATE).\n\n        This utilizes the approach of choosing s values such that the\n        sum of all of the scaled f values is equal to 1.\n\n        Arguments:\n         - seq_pos -- The current position we are at in the sequence.\n         - previous_vars -- All of the forward or backward variables\n           calculated so far.\n\n        Returns:\n         - The calculated scaling variable for the sequence item.\n\n        '
        state_letters = self._mm.state_alphabet
        s_value = 0
        for main_state in state_letters:
            emission = self._mm.emission_prob[main_state, self._seq.emissions[seq_pos]]
            trans_and_var_sum = 0
            for second_state in self._mm.transitions_from(main_state):
                var_value = previous_vars[second_state, seq_pos - 1]
                trans_value = self._mm.transition_prob[second_state, main_state]
                trans_and_var_sum += var_value * trans_value
            s_value += emission * trans_and_var_sum
        return s_value

    def _forward_recursion(self, cur_state, sequence_pos, forward_vars):
        if False:
            return 10
        'Calculate the value of the forward recursion (PRIVATE).\n\n        Arguments:\n         - cur_state -- The letter of the state we are calculating the\n           forward variable for.\n         - sequence_pos -- The position we are at in the training seq.\n         - forward_vars -- The current set of forward variables\n\n        '
        if sequence_pos not in self._s_values:
            self._s_values[sequence_pos] = self._calculate_s_value(sequence_pos, forward_vars)
        seq_letter = self._seq.emissions[sequence_pos]
        cur_emission_prob = self._mm.emission_prob[cur_state, seq_letter]
        scale_emission_prob = cur_emission_prob / self._s_values[sequence_pos]
        state_pos_sum = 0
        have_transition = 0
        for second_state in self._mm.transitions_from(cur_state):
            have_transition = 1
            prev_forward = forward_vars[second_state, sequence_pos - 1]
            cur_trans_prob = self._mm.transition_prob[second_state, cur_state]
            state_pos_sum += prev_forward * cur_trans_prob
        if have_transition:
            return scale_emission_prob * state_pos_sum
        else:
            return None

    def _backward_recursion(self, cur_state, sequence_pos, backward_vars):
        if False:
            for i in range(10):
                print('nop')
        'Calculate the value of the backward recursion (PRIVATE).\n\n        Arguments:\n         - cur_state -- The letter of the state we are calculating the\n           forward variable for.\n         - sequence_pos -- The position we are at in the training seq.\n         - backward_vars -- The current set of backward variables\n\n        '
        if sequence_pos not in self._s_values:
            self._s_values[sequence_pos] = self._calculate_s_value(sequence_pos, backward_vars)
        state_pos_sum = 0
        have_transition = 0
        for second_state in self._mm.transitions_from(cur_state):
            have_transition = 1
            seq_letter = self._seq.emissions[sequence_pos + 1]
            cur_emission_prob = self._mm.emission_prob[cur_state, seq_letter]
            prev_backward = backward_vars[second_state, sequence_pos + 1]
            cur_transition_prob = self._mm.transition_prob[cur_state, second_state]
            state_pos_sum += cur_emission_prob * prev_backward * cur_transition_prob
        if have_transition:
            return state_pos_sum / self._s_values[sequence_pos]
        else:
            return None

class LogDPAlgorithms(AbstractDPAlgorithms):
    """Implement forward and backward algorithms using a log approach.

    This uses the approach of calculating the sum of log probabilities
    using a lookup table for common values.

    XXX This is not implemented yet!
    """

    def __init__(self, markov_model, sequence):
        if False:
            for i in range(10):
                print('nop')
        'Initialize the class.'
        raise NotImplementedError("Haven't coded this yet...")