"""
Conditional random field
"""
from typing import List, Tuple, Dict, Union
import torch
from allennlp.common.checks import ConfigurationError
import allennlp.nn.util as util
VITERBI_DECODING = Tuple[List[int], float]

def allowed_transitions(constraint_type: str, labels: Dict[int, str]) -> List[Tuple[int, int]]:
    if False:
        print('Hello World!')
    '\n    Given labels and a constraint type, returns the allowed transitions. It will\n    additionally include transitions for the start and end states, which are used\n    by the conditional random field.\n\n    # Parameters\n\n    constraint_type : `str`, required\n        Indicates which constraint to apply. Current choices are\n        "BIO", "IOB1", "BIOUL", and "BMES".\n    labels : `Dict[int, str]`, required\n        A mapping {label_id -> label}. Most commonly this would be the value from\n        Vocabulary.get_index_to_token_vocabulary()\n\n    # Returns\n\n    `List[Tuple[int, int]]`\n        The allowed transitions (from_label_id, to_label_id).\n    '
    num_labels = len(labels)
    start_tag = num_labels
    end_tag = num_labels + 1
    labels_with_boundaries = list(labels.items()) + [(start_tag, 'START'), (end_tag, 'END')]
    allowed = []
    for (from_label_index, from_label) in labels_with_boundaries:
        if from_label in ('START', 'END'):
            from_tag = from_label
            from_entity = ''
        else:
            from_tag = from_label[0]
            from_entity = from_label[1:]
        for (to_label_index, to_label) in labels_with_boundaries:
            if to_label in ('START', 'END'):
                to_tag = to_label
                to_entity = ''
            else:
                to_tag = to_label[0]
                to_entity = to_label[1:]
            if is_transition_allowed(constraint_type, from_tag, from_entity, to_tag, to_entity):
                allowed.append((from_label_index, to_label_index))
    return allowed

def is_transition_allowed(constraint_type: str, from_tag: str, from_entity: str, to_tag: str, to_entity: str):
    if False:
        for i in range(10):
            print('nop')
    '\n    Given a constraint type and strings `from_tag` and `to_tag` that\n    represent the origin and destination of the transition, return whether\n    the transition is allowed under the given constraint type.\n\n    # Parameters\n\n    constraint_type : `str`, required\n        Indicates which constraint to apply. Current choices are\n        "BIO", "IOB1", "BIOUL", and "BMES".\n    from_tag : `str`, required\n        The tag that the transition originates from. For example, if the\n        label is `I-PER`, the `from_tag` is `I`.\n    from_entity : `str`, required\n        The entity corresponding to the `from_tag`. For example, if the\n        label is `I-PER`, the `from_entity` is `PER`.\n    to_tag : `str`, required\n        The tag that the transition leads to. For example, if the\n        label is `I-PER`, the `to_tag` is `I`.\n    to_entity : `str`, required\n        The entity corresponding to the `to_tag`. For example, if the\n        label is `I-PER`, the `to_entity` is `PER`.\n\n    # Returns\n\n    `bool`\n        Whether the transition is allowed under the given `constraint_type`.\n    '
    if to_tag == 'START' or from_tag == 'END':
        return False
    if constraint_type == 'BIOUL':
        if from_tag == 'START':
            return to_tag in ('O', 'B', 'U')
        if to_tag == 'END':
            return from_tag in ('O', 'L', 'U')
        return any([from_tag in ('O', 'L', 'U') and to_tag in ('O', 'B', 'U'), from_tag in ('B', 'I') and to_tag in ('I', 'L') and (from_entity == to_entity)])
    elif constraint_type == 'BIO':
        if from_tag == 'START':
            return to_tag in ('O', 'B')
        if to_tag == 'END':
            return from_tag in ('O', 'B', 'I')
        return any([to_tag in ('O', 'B'), to_tag == 'I' and from_tag in ('B', 'I') and (from_entity == to_entity)])
    elif constraint_type == 'IOB1':
        if from_tag == 'START':
            return to_tag in ('O', 'I')
        if to_tag == 'END':
            return from_tag in ('O', 'B', 'I')
        return any([to_tag in ('O', 'I'), to_tag == 'B' and from_tag in ('B', 'I') and (from_entity == to_entity)])
    elif constraint_type == 'BMES':
        if from_tag == 'START':
            return to_tag in ('B', 'S')
        if to_tag == 'END':
            return from_tag in ('E', 'S')
        return any([to_tag in ('B', 'S') and from_tag in ('E', 'S'), to_tag == 'M' and from_tag in ('B', 'M') and (from_entity == to_entity), to_tag == 'E' and from_tag in ('B', 'M') and (from_entity == to_entity)])
    else:
        raise ConfigurationError(f'Unknown constraint type: {constraint_type}')

class ConditionalRandomField(torch.nn.Module):
    """
    This module uses the "forward-backward" algorithm to compute
    the log-likelihood of its inputs assuming a conditional random field model.

    See, e.g. http://www.cs.columbia.edu/~mcollins/fb.pdf

    # Parameters

    num_tags : `int`, required
        The number of tags.
    constraints : `List[Tuple[int, int]]`, optional (default = `None`)
        An optional list of allowed transitions (from_tag_id, to_tag_id).
        These are applied to `viterbi_tags()` but do not affect `forward()`.
        These should be derived from `allowed_transitions` so that the
        start and end transitions are handled correctly for your tag type.
    include_start_end_transitions : `bool`, optional (default = `True`)
        Whether to include the start and end transition parameters.
    """

    def __init__(self, num_tags: int, constraints: List[Tuple[int, int]]=None, include_start_end_transitions: bool=True) -> None:
        if False:
            return 10
        super().__init__()
        self.num_tags = num_tags
        self.transitions = torch.nn.Parameter(torch.empty(num_tags, num_tags))
        if constraints is None:
            constraint_mask = torch.full((num_tags + 2, num_tags + 2), 1.0)
        else:
            constraint_mask = torch.full((num_tags + 2, num_tags + 2), 0.0)
            for (i, j) in constraints:
                constraint_mask[i, j] = 1.0
        self._constraint_mask = torch.nn.Parameter(constraint_mask, requires_grad=False)
        self.include_start_end_transitions = include_start_end_transitions
        if include_start_end_transitions:
            self.start_transitions = torch.nn.Parameter(torch.Tensor(num_tags))
            self.end_transitions = torch.nn.Parameter(torch.Tensor(num_tags))
        self.reset_parameters()

    def reset_parameters(self):
        if False:
            return 10
        torch.nn.init.xavier_normal_(self.transitions)
        if self.include_start_end_transitions:
            torch.nn.init.normal_(self.start_transitions)
            torch.nn.init.normal_(self.end_transitions)

    def _input_likelihood(self, logits: torch.Tensor, transitions: torch.Tensor, mask: torch.BoolTensor) -> torch.Tensor:
        if False:
            return 10
        'Computes the (batch_size,) denominator term $Z(x)$, per example, for the log-likelihood\n\n        This is the sum of the likelihoods across all possible state sequences.\n\n        Args:\n            logits (torch.Tensor): a (batch_size, sequence_length num_tags) tensor of\n                unnormalized log-probabilities\n            transitions (torch.Tensor): a (batch_size, num_tags, num_tags) tensor of transition scores\n            mask (torch.BoolTensor): a (batch_size, sequence_length) tensor of masking flags\n\n        Returns:\n            torch.Tensor: (batch_size,) denominator term $Z(x)$, per example, for the log-likelihood\n        '
        (batch_size, sequence_length, num_tags) = logits.size()
        mask = mask.transpose(0, 1).contiguous()
        logits = logits.transpose(0, 1).contiguous()
        if self.include_start_end_transitions:
            alpha = self.start_transitions.view(1, num_tags) + logits[0]
        else:
            alpha = logits[0]
        for i in range(1, sequence_length):
            emit_scores = logits[i].view(batch_size, 1, num_tags)
            transition_scores = transitions.view(1, num_tags, num_tags)
            broadcast_alpha = alpha.view(batch_size, num_tags, 1)
            inner = broadcast_alpha + emit_scores + transition_scores
            alpha = util.logsumexp(inner, 1) * mask[i].view(batch_size, 1) + alpha * (~mask[i]).view(batch_size, 1)
        if self.include_start_end_transitions:
            stops = alpha + self.end_transitions.view(1, num_tags)
        else:
            stops = alpha
        return util.logsumexp(stops)

    def _joint_likelihood(self, logits: torch.Tensor, transitions: torch.Tensor, tags: torch.Tensor, mask: torch.BoolTensor) -> torch.Tensor:
        if False:
            print('Hello World!')
        'Computes the numerator term for the log-likelihood, which is just score(inputs, tags)\n\n        Args:\n            logits (torch.Tensor): a (batch_size, sequence_length num_tags) tensor of unnormalized\n                log-probabilities\n            transitions (torch.Tensor): a (batch_size, num_tags, num_tags) tensor of transition scores\n            tags (torch.Tensor): output tag sequences (batch_size, sequence_length) $y$ for each input sequence\n            mask (torch.BoolTensor): a (batch_size, sequence_length) tensor of masking flags\n\n        Returns:\n            torch.Tensor: numerator term for the log-likelihood, which is just score(inputs, tags)\n        '
        (batch_size, sequence_length, _) = logits.data.shape
        logits = logits.transpose(0, 1).contiguous()
        mask = mask.transpose(0, 1).contiguous()
        tags = tags.transpose(0, 1).contiguous()
        if self.include_start_end_transitions:
            score = self.start_transitions.index_select(0, tags[0])
        else:
            score = 0.0
        for i in range(sequence_length - 1):
            (current_tag, next_tag) = (tags[i], tags[i + 1])
            transition_score = transitions[current_tag.view(-1), next_tag.view(-1)]
            emit_score = logits[i].gather(1, current_tag.view(batch_size, 1)).squeeze(1)
            score = score + transition_score * mask[i + 1] + emit_score * mask[i]
        last_tag_index = mask.sum(0).long() - 1
        last_tags = tags.gather(0, last_tag_index.view(1, batch_size)).squeeze(0)
        if self.include_start_end_transitions:
            last_transition_score = self.end_transitions.index_select(0, last_tags)
        else:
            last_transition_score = 0.0
        last_inputs = logits[-1]
        last_input_score = last_inputs.gather(1, last_tags.view(-1, 1))
        last_input_score = last_input_score.squeeze()
        score = score + last_transition_score + last_input_score * mask[-1]
        return score

    def forward(self, inputs: torch.Tensor, tags: torch.Tensor, mask: torch.BoolTensor=None) -> torch.Tensor:
        if False:
            print('Hello World!')
        'Computes the log likelihood for the given batch of input sequences $(x,y)$\n\n        Args:\n            inputs (torch.Tensor): (batch_size, sequence_length, num_tags) tensor of logits for the inputs $x$\n            tags (torch.Tensor): (batch_size, sequence_length) tensor of tags $y$\n            mask (torch.BoolTensor, optional): (batch_size, sequence_length) tensor of masking flags.\n                Defaults to None.\n\n        Returns:\n            torch.Tensor: (batch_size,) log likelihoods $log P(y|x)$ for each input\n        '
        if mask is None:
            mask = torch.ones(*tags.size(), dtype=torch.bool, device=inputs.device)
        else:
            mask = mask.to(torch.bool)
        log_denominator = self._input_likelihood(inputs, self.transitions, mask)
        log_numerator = self._joint_likelihood(inputs, self.transitions, tags, mask)
        return torch.sum(log_numerator - log_denominator)

    def viterbi_tags(self, logits: torch.Tensor, mask: torch.BoolTensor=None, top_k: int=None) -> Union[List[VITERBI_DECODING], List[List[VITERBI_DECODING]]]:
        if False:
            while True:
                i = 10
        '\n        Uses viterbi algorithm to find most likely tags for the given inputs.\n        If constraints are applied, disallows all other transitions.\n\n        Returns a list of results, of the same size as the batch (one result per batch member)\n        Each result is a List of length top_k, containing the top K viterbi decodings\n        Each decoding is a tuple  (tag_sequence, viterbi_score)\n\n        For backwards compatibility, if top_k is None, then instead returns a flat list of\n        tag sequences (the top tag sequence for each batch item).\n        '
        if mask is None:
            mask = torch.ones(*logits.shape[:2], dtype=torch.bool, device=logits.device)
        if top_k is None:
            top_k = 1
            flatten_output = True
        else:
            flatten_output = False
        (_, max_seq_length, num_tags) = logits.size()
        (logits, mask) = (logits.data, mask.data)
        start_tag = num_tags
        end_tag = num_tags + 1
        transitions = torch.full((num_tags + 2, num_tags + 2), -10000.0, device=logits.device)
        constrained_transitions = self.transitions * self._constraint_mask[:num_tags, :num_tags] + -10000.0 * (1 - self._constraint_mask[:num_tags, :num_tags])
        transitions[:num_tags, :num_tags] = constrained_transitions.data
        if self.include_start_end_transitions:
            transitions[start_tag, :num_tags] = self.start_transitions.detach() * self._constraint_mask[start_tag, :num_tags].data + -10000.0 * (1 - self._constraint_mask[start_tag, :num_tags].detach())
            transitions[:num_tags, end_tag] = self.end_transitions.detach() * self._constraint_mask[:num_tags, end_tag].data + -10000.0 * (1 - self._constraint_mask[:num_tags, end_tag].detach())
        else:
            transitions[start_tag, :num_tags] = -10000.0 * (1 - self._constraint_mask[start_tag, :num_tags].detach())
            transitions[:num_tags, end_tag] = -10000.0 * (1 - self._constraint_mask[:num_tags, end_tag].detach())
        best_paths = []
        tag_sequence = torch.empty(max_seq_length + 2, num_tags + 2, device=logits.device)
        for (prediction, prediction_mask) in zip(logits, mask):
            mask_indices = prediction_mask.nonzero(as_tuple=False).squeeze()
            masked_prediction = torch.index_select(prediction, 0, mask_indices)
            sequence_length = masked_prediction.shape[0]
            tag_sequence.fill_(-10000.0)
            tag_sequence[0, start_tag] = 0.0
            tag_sequence[1:sequence_length + 1, :num_tags] = masked_prediction
            tag_sequence[sequence_length + 1, end_tag] = 0.0
            (viterbi_paths, viterbi_scores) = util.viterbi_decode(tag_sequence=tag_sequence[:sequence_length + 2], transition_matrix=transitions, top_k=top_k)
            top_k_paths = []
            for (viterbi_path, viterbi_score) in zip(viterbi_paths, viterbi_scores):
                viterbi_path = viterbi_path[1:-1]
                top_k_paths.append((viterbi_path, viterbi_score.item()))
            best_paths.append(top_k_paths)
        if flatten_output:
            return [top_k_paths[0] for top_k_paths in best_paths]
        return best_paths