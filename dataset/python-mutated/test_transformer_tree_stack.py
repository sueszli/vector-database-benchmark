import pytest
import torch
from stanza.models.constituency.transformer_tree_stack import TransformerTreeStack
pytestmark = [pytest.mark.pipeline, pytest.mark.travis]

def test_initial_state():
    if False:
        while True:
            i = 10
    '\n    Test that the initial state has the expected shapes\n    '
    ts = TransformerTreeStack(3, 5, 0.0)
    initial = ts.initial_state()
    assert len(initial) == 1
    assert initial.value.output.shape == torch.Size([5])
    assert initial.value.key_stack.shape == torch.Size([1, 5])
    assert initial.value.value_stack.shape == torch.Size([1, 5])

def test_output():
    if False:
        return 10
    '\n    Test that you can get an expected output shape from the TTS\n    '
    ts = TransformerTreeStack(3, 5, 0.0)
    initial = ts.initial_state()
    out = ts.output(initial)
    assert out.shape == torch.Size([5])
    assert torch.allclose(initial.value.output, out)

def test_push_state_single():
    if False:
        while True:
            i = 10
    '\n    Test that stacks are being updated correctly when using a single stack\n\n    Values of the attention are not verified, though\n    '
    ts = TransformerTreeStack(3, 5, 0.0)
    initial = ts.initial_state()
    rand_input = torch.randn(1, 3)
    stacks = ts.push_states([initial], ['A'], rand_input)
    stacks = ts.push_states(stacks, ['B'], rand_input)
    assert len(stacks) == 1
    assert len(stacks[0]) == 3
    assert stacks[0].value.value == 'B'
    assert stacks[0].pop().value.value == 'A'
    assert stacks[0].pop().pop().value.value is None

def test_push_state_same_length():
    if False:
        print('Hello World!')
    '\n    Test that stacks are being updated correctly when using 3 stacks of the same length\n\n    Values of the attention are not verified, though\n    '
    ts = TransformerTreeStack(3, 5, 0.0)
    initial = ts.initial_state()
    rand_input = torch.randn(3, 3)
    stacks = ts.push_states([initial, initial, initial], ['A', 'A', 'A'], rand_input)
    stacks = ts.push_states(stacks, ['B', 'B', 'B'], rand_input)
    stacks = ts.push_states(stacks, ['C', 'C', 'C'], rand_input)
    assert len(stacks) == 3
    for s in stacks:
        assert len(s) == 4
        assert s.value.key_stack.shape == torch.Size([4, 5])
        assert s.value.value_stack.shape == torch.Size([4, 5])
        assert s.value.value == 'C'
        assert s.pop().value.value == 'B'
        assert s.pop().pop().value.value == 'A'
        assert s.pop().pop().pop().value.value is None

def test_push_state_different_length():
    if False:
        for i in range(10):
            print('nop')
    '\n    Test what happens if stacks of different lengths are passed in\n    '
    ts = TransformerTreeStack(3, 5, 0.0)
    initial = ts.initial_state()
    rand_input = torch.randn(2, 3)
    one_step = ts.push_states([initial], ['A'], rand_input[0:1, :])[0]
    stacks = [one_step, initial]
    stacks = ts.push_states(stacks, ['B', 'C'], rand_input)
    assert len(stacks) == 2
    assert len(stacks[0]) == 3
    assert len(stacks[1]) == 2
    assert stacks[0].pop().value.value == 'A'
    assert stacks[0].value.value == 'B'
    assert stacks[1].value.value == 'C'
    assert stacks[0].value.key_stack.shape == torch.Size([3, 5])
    assert stacks[1].value.key_stack.shape == torch.Size([2, 5])

def test_mask():
    if False:
        for i in range(10):
            print('nop')
    '\n    Test that a mask prevents the softmax from picking up unwanted values\n    '
    ts = TransformerTreeStack(3, 5, 0.0)
    random_v = torch.tensor([[[0.1, 0.2, 0.3, 0.4, 0.5]]])
    double_v = random_v * 2
    value = torch.cat([random_v, double_v], axis=1)
    random_k = torch.randn(1, 1, 5)
    key = torch.cat([random_k, random_k], axis=1)
    query = torch.randn(1, 5)
    output = ts.attention(key, query, value)
    expected_output = (random_v + double_v) / 2
    assert torch.allclose(output, expected_output)
    mask = torch.zeros(1, 2, dtype=torch.bool)
    mask[0][0] = True
    output = ts.attention(key, query, value, mask)
    assert torch.allclose(output, double_v)
    mask = torch.zeros(1, 2, dtype=torch.bool)
    mask[0][1] = True
    output = ts.attention(key, query, value, mask)
    assert torch.allclose(output, random_v)

def test_position():
    if False:
        print('Hello World!')
    '\n    Test that nothing goes horribly wrong when position encodings are used\n\n    Does not actually test the results of the encodings\n    '
    ts = TransformerTreeStack(4, 5, 0.0, use_position=True)
    initial = ts.initial_state()
    assert len(initial) == 1
    assert initial.value.output.shape == torch.Size([5])
    assert initial.value.key_stack.shape == torch.Size([1, 5])
    assert initial.value.value_stack.shape == torch.Size([1, 5])
    rand_input = torch.randn(2, 4)
    one_step = ts.push_states([initial], ['A'], rand_input[0:1, :])[0]
    stacks = [one_step, initial]
    stacks = ts.push_states(stacks, ['B', 'C'], rand_input)

def test_length_limit():
    if False:
        print('Hello World!')
    '\n    Test that the length limit drops nodes as the length limit is exceeded\n    '
    ts = TransformerTreeStack(4, 5, 0.0, length_limit=2)
    initial = ts.initial_state()
    assert len(initial) == 1
    assert initial.value.output.shape == torch.Size([5])
    assert initial.value.key_stack.shape == torch.Size([1, 5])
    assert initial.value.value_stack.shape == torch.Size([1, 5])
    data = torch.tensor([[0.1, 0.2, 0.3, 0.4]])
    stacks = ts.push_states([initial], ['A'], data)
    stacks = ts.push_states(stacks, ['B'], data)
    assert len(stacks) == 1
    assert len(stacks[0]) == 3
    assert stacks[0].value.key_stack.shape[0] == 3
    assert stacks[0].value.value_stack.shape[0] == 3
    stacks = ts.push_states(stacks, ['C'], data)
    assert len(stacks) == 1
    assert len(stacks[0]) == 4
    assert stacks[0].value.key_stack.shape[0] == 3
    assert stacks[0].value.value_stack.shape[0] == 3
    stacks = ts.push_states(stacks, ['D'], data)
    assert len(stacks) == 1
    assert len(stacks[0]) == 5
    assert stacks[0].value.key_stack.shape[0] == 3
    assert stacks[0].value.value_stack.shape[0] == 3

def test_two_heads():
    if False:
        for i in range(10):
            print('nop')
    '\n    Test that the length limit drops nodes as the length limit is exceeded\n    '
    ts = TransformerTreeStack(4, 6, 0.0, num_heads=2)
    initial = ts.initial_state()
    assert len(initial) == 1
    assert initial.value.output.shape == torch.Size([6])
    assert initial.value.key_stack.shape == torch.Size([1, 6])
    assert initial.value.value_stack.shape == torch.Size([1, 6])
    rand_input = torch.randn(2, 4)
    one_step = ts.push_states([initial], ['A'], rand_input[0:1, :])[0]
    stacks = [one_step, initial]
    stacks = ts.push_states(stacks, ['B', 'C'], rand_input)
    assert len(stacks) == 2
    assert len(stacks[0]) == 3
    assert len(stacks[1]) == 2
    assert stacks[0].pop().value.value == 'A'
    assert stacks[0].value.value == 'B'
    assert stacks[1].value.value == 'C'
    assert stacks[0].value.key_stack.shape == torch.Size([3, 6])
    assert stacks[1].value.key_stack.shape == torch.Size([2, 6])