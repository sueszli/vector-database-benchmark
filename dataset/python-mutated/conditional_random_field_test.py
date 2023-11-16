import itertools
import math
from pytest import approx, raises
import torch
from numpy.testing import assert_allclose
from allennlp.modules import ConditionalRandomField
from allennlp.modules.conditional_random_field import ConditionalRandomFieldWeightEmission, ConditionalRandomFieldWeightTrans, ConditionalRandomFieldWeightLannoy
from allennlp.modules.conditional_random_field.conditional_random_field import allowed_transitions
from allennlp.common.checks import ConfigurationError
from allennlp.common.testing import AllenNlpTestCase

class TestConditionalRandomField(AllenNlpTestCase):

    def setup_method(self):
        if False:
            for i in range(10):
                print('nop')
        super().setup_method()
        self.logits = torch.Tensor([[[0, 0, 0.5, 0.5, 0.2], [0, 0, 0.3, 0.3, 0.1], [0, 0, 0.9, 10, 1]], [[0, 0, 0.2, 0.5, 0.2], [0, 0, 3, 0.3, 0.1], [0, 0, 0.9, 1, 1]]])
        self.tags = torch.LongTensor([[2, 3, 4], [3, 2, 2]])
        self.transitions = torch.Tensor([[0.1, 0.2, 0.3, 0.4, 0.5], [0.8, 0.3, 0.1, 0.7, 0.9], [-0.3, 2.1, -5.6, 3.4, 4.0], [0.2, 0.4, 0.6, -0.3, -0.4], [1.0, 1.0, 1.0, 1.0, 1.0]])
        self.transitions_from_start = torch.Tensor([0.1, 0.2, 0.3, 0.4, 0.6])
        self.transitions_to_end = torch.Tensor([-0.1, -0.2, 0.3, -0.4, -0.4])
        self.crf = ConditionalRandomField(5)
        self.crf.transitions = torch.nn.Parameter(self.transitions)
        self.crf.start_transitions = torch.nn.Parameter(self.transitions_from_start)
        self.crf.end_transitions = torch.nn.Parameter(self.transitions_to_end)

    def score(self, logits, tags):
        if False:
            print('Hello World!')
        '\n        Computes the likelihood score for the given sequence of tags,\n        given the provided logits (and the transition weights in the CRF model)\n        '
        total = self.transitions_from_start[tags[0]] + self.transitions_to_end[tags[-1]]
        for (tag, next_tag) in zip(tags, tags[1:]):
            total += self.transitions[tag, next_tag]
        for (logit, tag) in zip(logits, tags):
            total += logit[tag]
        return total

    def naive_most_likely_sequence(self, logits, mask):
        if False:
            print('Hello World!')
        most_likely_tags = []
        best_scores = []
        for (logit, mas) in zip(logits, mask):
            mask_indices = mas.nonzero(as_tuple=False).squeeze()
            logit = torch.index_select(logit, 0, mask_indices)
            sequence_length = logit.shape[0]
            (most_likely, most_likelihood) = (None, -float('inf'))
            for tags in itertools.product(range(5), repeat=sequence_length):
                score = self.score(logit.data, tags)
                if score > most_likelihood:
                    (most_likely, most_likelihood) = (tags, score)
            most_likely_tags.append(list(most_likely))
            best_scores.append(most_likelihood)
        return (most_likely_tags, best_scores)

    def test_forward_works_without_mask(self):
        if False:
            for i in range(10):
                print('nop')
        log_likelihood = self.crf(self.logits, self.tags).item()
        manual_log_likelihood = 0.0
        for (logits_i, tags_i) in zip(self.logits, self.tags):
            numerator = self.score(logits_i.detach(), tags_i.detach())
            all_scores = [self.score(logits_i.detach(), tags_j) for tags_j in itertools.product(range(5), repeat=3)]
            denominator = math.log(sum((math.exp(score) for score in all_scores)))
            manual_log_likelihood += numerator - denominator
        assert manual_log_likelihood.item() == approx(log_likelihood)

    def test_forward_works_with_mask(self):
        if False:
            return 10
        mask = torch.tensor([[True, True, True], [True, True, False]])
        log_likelihood = self.crf(self.logits, self.tags, mask).item()
        manual_log_likelihood = 0.0
        for (logits_i, tags_i, mask_i) in zip(self.logits, self.tags, mask):
            sequence_length = torch.sum(mask_i.detach())
            logits_i = logits_i.data[:sequence_length]
            tags_i = tags_i.data[:sequence_length]
            numerator = self.score(logits_i, tags_i)
            all_scores = [self.score(logits_i, tags_j) for tags_j in itertools.product(range(5), repeat=sequence_length)]
            denominator = math.log(sum((math.exp(score) for score in all_scores)))
            manual_log_likelihood += numerator - denominator
        assert manual_log_likelihood.item() == approx(log_likelihood)

    def test_viterbi_tags(self):
        if False:
            print('Hello World!')
        mask = torch.tensor([[True, True, True], [True, False, True]])
        viterbi_path = self.crf.viterbi_tags(self.logits, mask)
        viterbi_tags = [x for (x, y) in viterbi_path]
        viterbi_scores = [y for (x, y) in viterbi_path]
        (most_likely_tags, best_scores) = self.naive_most_likely_sequence(self.logits, mask)
        assert viterbi_tags == most_likely_tags
        assert_allclose(viterbi_scores, best_scores, rtol=1e-05)

    def test_viterbi_tags_no_mask(self):
        if False:
            print('Hello World!')
        viterbi_path = self.crf.viterbi_tags(self.logits)
        viterbi_tags = [x for (x, y) in viterbi_path]
        viterbi_scores = [y for (x, y) in viterbi_path]
        mask = torch.tensor([[True, True, True], [True, True, True]])
        (most_likely_tags, best_scores) = self.naive_most_likely_sequence(self.logits, mask)
        assert viterbi_tags == most_likely_tags
        assert_allclose(viterbi_scores, best_scores, rtol=1e-05)

    def test_viterbi_tags_top_k(self):
        if False:
            i = 10
            return i + 15
        mask = torch.tensor([[True, True, True], [True, True, False]])
        best_paths = self.crf.viterbi_tags(self.logits, mask, top_k=2)
        top_path_and_score = [top_k_paths[0] for top_k_paths in best_paths]
        assert top_path_and_score == self.crf.viterbi_tags(self.logits, mask)
        next_path_and_score = [top_k_paths[1] for top_k_paths in best_paths]
        next_viterbi_tags = [x for (x, _) in next_path_and_score]
        assert next_viterbi_tags == [[4, 2, 3], [3, 2]]

    def test_constrained_viterbi_tags(self):
        if False:
            return 10
        constraints = {(0, 0), (0, 1), (1, 1), (1, 2), (2, 2), (2, 3), (3, 3), (3, 4), (4, 4), (4, 0)}
        for i in range(5):
            constraints.add((5, i))
            constraints.add((i, 6))
        crf = ConditionalRandomField(num_tags=5, constraints=constraints)
        crf.transitions = torch.nn.Parameter(self.transitions)
        crf.start_transitions = torch.nn.Parameter(self.transitions_from_start)
        crf.end_transitions = torch.nn.Parameter(self.transitions_to_end)
        mask = torch.tensor([[True, True, True], [True, True, False]])
        viterbi_path = crf.viterbi_tags(self.logits, mask)
        viterbi_tags = [x for (x, y) in viterbi_path]
        assert viterbi_tags == [[2, 3, 3], [2, 3]]

    def test_allowed_transitions(self):
        if False:
            return 10
        bio_labels = ['O', 'B-X', 'I-X', 'B-Y', 'I-Y']
        allowed = allowed_transitions('BIO', dict(enumerate(bio_labels)))
        assert set(allowed) == {(0, 0), (0, 1), (0, 3), (0, 6), (1, 0), (1, 1), (1, 2), (1, 3), (1, 6), (2, 0), (2, 1), (2, 2), (2, 3), (2, 6), (3, 0), (3, 1), (3, 3), (3, 4), (3, 6), (4, 0), (4, 1), (4, 3), (4, 4), (4, 6), (5, 0), (5, 1), (5, 3)}
        bioul_labels = ['O', 'B-X', 'I-X', 'L-X', 'U-X', 'B-Y', 'I-Y', 'L-Y', 'U-Y']
        allowed = allowed_transitions('BIOUL', dict(enumerate(bioul_labels)))
        assert set(allowed) == {(0, 0), (0, 1), (0, 4), (0, 5), (0, 8), (0, 10), (1, 2), (1, 3), (2, 2), (2, 3), (3, 0), (3, 1), (3, 4), (3, 5), (3, 8), (3, 10), (4, 0), (4, 1), (4, 4), (4, 5), (4, 8), (4, 10), (5, 6), (5, 7), (6, 6), (6, 7), (7, 0), (7, 1), (7, 4), (7, 5), (7, 8), (7, 10), (8, 0), (8, 1), (8, 4), (8, 5), (8, 8), (8, 10), (9, 0), (9, 1), (9, 4), (9, 5), (9, 8)}
        iob1_labels = ['O', 'B-X', 'I-X', 'B-Y', 'I-Y']
        allowed = allowed_transitions('IOB1', dict(enumerate(iob1_labels)))
        assert set(allowed) == {(0, 0), (0, 2), (0, 4), (0, 6), (1, 0), (1, 1), (1, 2), (1, 4), (1, 6), (2, 0), (2, 1), (2, 2), (2, 4), (2, 6), (3, 0), (3, 2), (3, 3), (3, 4), (3, 6), (4, 0), (4, 2), (4, 3), (4, 4), (4, 6), (5, 0), (5, 2), (5, 4)}
        with raises(ConfigurationError):
            allowed_transitions('allennlp', {})
        bmes_labels = ['B-X', 'M-X', 'E-X', 'S-X', 'B-Y', 'M-Y', 'E-Y', 'S-Y']
        allowed = allowed_transitions('BMES', dict(enumerate(bmes_labels)))
        assert set(allowed) == {(0, 1), (0, 2), (1, 1), (1, 2), (2, 0), (2, 3), (2, 4), (2, 7), (2, 9), (3, 0), (3, 3), (3, 4), (3, 7), (3, 9), (4, 5), (4, 6), (5, 5), (5, 6), (6, 0), (6, 3), (6, 4), (6, 7), (6, 9), (7, 0), (7, 3), (7, 4), (7, 7), (7, 9), (8, 0), (8, 3), (8, 4), (8, 7)}

class TestConditionalRandomFieldWeightEmission(TestConditionalRandomField):

    def setup_method(self):
        if False:
            return 10
        super().setup_method()
        self.label_weights = torch.FloatTensor([1.0, 1.0, 0.5, 0.5, 0.5])
        self.crf = ConditionalRandomFieldWeightEmission(5, label_weights=self.label_weights)
        self.crf.transitions = torch.nn.Parameter(self.transitions)
        self.crf.start_transitions = torch.nn.Parameter(self.transitions_from_start)
        self.crf.end_transitions = torch.nn.Parameter(self.transitions_to_end)
        self.crf.label_weights = torch.nn.Parameter(self.label_weights, requires_grad=False)

    def score_with_weights(self, logits, tags):
        if False:
            return 10
        '\n        Computes the likelihood score for the given sequence of tags,\n        given the provided logits, the transition weights in the CRF model\n        and the label weights.\n        '
        total = self.transitions_from_start[tags[0]] + self.transitions_to_end[tags[-1]]
        for (tag, next_tag) in zip(tags, tags[1:]):
            total += self.transitions[tag, next_tag]
        for (logit, tag) in zip(logits, tags):
            total += logit[tag] * self.label_weights[tag]
        return total

    def test_forward_works_without_mask(self):
        if False:
            print('Hello World!')
        log_likelihood = self.crf(self.logits, self.tags).item()
        manual_log_likelihood = 0.0
        for (logits_i, tags_i) in zip(self.logits, self.tags):
            numerator = self.score_with_weights(logits_i.detach(), tags_i.detach())
            all_scores = [self.score_with_weights(logits_i.detach(), tags_j) for tags_j in itertools.product(range(5), repeat=3)]
            denominator = math.log(sum((math.exp(score) for score in all_scores)))
            manual_log_likelihood += numerator - denominator
        assert manual_log_likelihood.item() == approx(log_likelihood)

    def test_forward_works_with_mask(self):
        if False:
            i = 10
            return i + 15
        mask = torch.tensor([[True, True, True], [True, True, False]])
        log_likelihood = self.crf(self.logits, self.tags, mask).item()
        manual_log_likelihood = 0.0
        for (logits_i, tags_i, mask_i) in zip(self.logits, self.tags, mask):
            sequence_length = torch.sum(mask_i.detach())
            logits_i = logits_i.data[:sequence_length]
            tags_i = tags_i.data[:sequence_length]
            numerator = self.score_with_weights(logits_i, tags_i)
            all_scores = [self.score_with_weights(logits_i, tags_j) for tags_j in itertools.product(range(5), repeat=sequence_length)]
            denominator = math.log(sum((math.exp(score) for score in all_scores)))
            manual_log_likelihood += numerator - denominator
        assert manual_log_likelihood.item() == approx(log_likelihood)

class TestConditionalRandomFieldWeightTrans(TestConditionalRandomFieldWeightEmission):

    def setup_method(self):
        if False:
            return 10
        super().setup_method()
        self.label_weights = torch.FloatTensor([1.0, 1.0, 0.5, 0.5, 0.5])
        self.crf = ConditionalRandomFieldWeightTrans(5, label_weights=self.label_weights)
        self.crf.transitions = torch.nn.Parameter(self.transitions)
        self.crf.start_transitions = torch.nn.Parameter(self.transitions_from_start)
        self.crf.end_transitions = torch.nn.Parameter(self.transitions_to_end)
        self.crf.label_weights = torch.nn.Parameter(self.label_weights, requires_grad=False)

    def score_with_weights(self, logits, tags):
        if False:
            i = 10
            return i + 15
        '\n        Computes the likelihood score for the given sequence of tags,\n        given the provided logits, the transition weights in the CRF model\n        and the label weights.\n        '
        total = self.transitions_from_start[tags[0]] + self.transitions_to_end[tags[-1]]
        for (tag, next_tag) in zip(tags, tags[1:]):
            total += self.transitions[tag, next_tag] * self.label_weights[tag]
        for (logit, tag) in zip(logits, tags):
            total += logit[tag] * self.label_weights[tag]
        return total

class TestConditionalRandomFieldWeightLannoy(TestConditionalRandomFieldWeightEmission):

    def setup_method(self):
        if False:
            for i in range(10):
                print('nop')
        super().setup_method()
        self.label_weights = torch.FloatTensor([1.0, 1.0, 1.0, 1.0, 1.0])
        self.crf = ConditionalRandomFieldWeightLannoy(5, label_weights=self.label_weights)
        self.crf.transitions = torch.nn.Parameter(self.transitions)
        self.crf.start_transitions = torch.nn.Parameter(self.transitions_from_start)
        self.crf.end_transitions = torch.nn.Parameter(self.transitions_to_end)
        self.crf.label_weights = torch.nn.Parameter(self.label_weights, requires_grad=False)