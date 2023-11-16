import pytest
import torch
from catalyst.metrics.functional import f1_score, fbeta_score, precision, precision_recall_fbeta_support, recall

def test_precision_recall_f_binary_single_class() -> None:
    if False:
        for i in range(10):
            print('nop')
    'Metrics test'
    assert 1.0 == precision([1, 1], [1, 1])[1]
    assert 1.0 == recall([1, 1], [1, 1])[1]
    assert 1.0 == f1_score([1, 1], [1, 1])[1]
    assert 1.0 == fbeta_score([1, 1], [1, 1], 0)[1]
    assert 3.0 == f1_score([0, 1, 2], [0, 1, 2]).sum().item()
    assert 3.0 == precision([0, 1, 2], [0, 1, 2]).sum().item()
    assert 3.0 == recall([0, 1, 2], [0, 1, 2]).sum().item()

@pytest.mark.parametrize(['outputs', 'targets', 'precision_true', 'recall_true', 'fbeta_true', 'support_true'], [pytest.param(torch.tensor([[0, 0, 1, 1, 0, 1, 0, 1]]), torch.tensor([[0, 1, 0, 1, 0, 0, 1, 1]]), 0.5, 0.5, 0.5, 4)])
def test_precision_recall_fbeta_support_binary(outputs, targets, precision_true, recall_true, fbeta_true, support_true) -> None:
    if False:
        for i in range(10):
            print('nop')
    '\n    Test for precision_recall_fbeta_support.\n\n    Args:\n        outputs: test arg\n        targets: test arg\n        precision_true: test arg\n        recall_true: test arg\n        fbeta_true: test arg\n        support_true: test arg\n    '
    (precision_score, recall_score, fbeta_score_value, support) = precision_recall_fbeta_support(outputs=outputs, targets=targets)
    assert torch.isclose(precision_score[1], torch.tensor(precision_true))
    assert torch.isclose(recall_score[1], torch.tensor(recall_true))
    assert torch.isclose(fbeta_score_value[1], torch.tensor(fbeta_true))
    assert support[1] == support_true