"""
Based on

Transition-based Parsing with Stack-Transformers
Ramon Fernandez Astudillo, Miguel Ballesteros, Tahira Naseem,
  Austin Blodget, and Radu Florian
https://aclanthology.org/2020.findings-emnlp.89.pdf
"""
from collections import namedtuple
import torch
import torch.nn as nn
from stanza.models.constituency.positional_encoding import SinusoidalEncoding
from stanza.models.constituency.tree_stack import TreeStack
Node = namedtuple('Node', ['value', 'key_stack', 'value_stack', 'output'])

class TransformerTreeStack(nn.Module):

    def __init__(self, input_size, output_size, input_dropout, length_limit=None, use_position=False, num_heads=1):
        if False:
            for i in range(10):
                print('nop')
        '\n        Builds the internal matrices and start parameter\n\n        TODO: currently only one attention head, implement MHA\n        '
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.inv_sqrt_output_size = 1 / output_size ** 0.5
        self.num_heads = num_heads
        self.w_query = nn.Linear(input_size, output_size)
        self.w_key = nn.Linear(input_size, output_size)
        self.w_value = nn.Linear(input_size, output_size)
        self.register_parameter('start_embedding', torch.nn.Parameter(0.2 * torch.randn(input_size, requires_grad=True)))
        if isinstance(input_dropout, nn.Module):
            self.input_dropout = input_dropout
        else:
            self.input_dropout = nn.Dropout(input_dropout)
        if length_limit is not None and length_limit < 1:
            raise ValueError('length_limit < 1 makes no sense')
        self.length_limit = length_limit
        self.use_position = use_position
        if use_position:
            self.position_encoding = SinusoidalEncoding(model_dim=self.input_size, max_len=512)

    def attention(self, key, query, value, mask=None):
        if False:
            i = 10
            return i + 15
        '\n        Calculate attention for the given key, query value\n\n        Where B is the number of items stacked together, N is the length:\n        The key should be BxNxD\n        The query is BxD\n        The value is BxNxD\n\n        If mask is specified, it should be BxN of True/False values,\n        where True means that location is masked out\n\n        Reshapes and reorders are used to handle num_heads\n\n        Return will be softmax(query x key^T) * value\n        of size BxD\n        '
        B = key.shape[0]
        N = key.shape[1]
        D = key.shape[2]
        H = self.num_heads
        query = query.unsqueeze(2)
        query = query.reshape((B, H, -1, 1))
        key = key.reshape((B, N, H, -1))
        key = key.transpose(1, 2)
        value = value.reshape((B, N, H, -1))
        value = value.transpose(1, 2)
        attn = torch.matmul(key, query).squeeze(3) * self.inv_sqrt_output_size
        if mask is not None:
            mask = mask.unsqueeze(1)
            mask = mask.expand(-1, H, -1)
            attn.masked_fill_(mask, float('-inf'))
        attn = torch.softmax(attn, dim=2).unsqueeze(2)
        output = torch.matmul(attn, value).squeeze(2)
        output = output.reshape(B, -1)
        return output

    def initial_state(self, initial_value=None):
        if False:
            return 10
        '\n        Return an initial state based on a single layer of attention\n\n        Running attention might be overkill, but it is the simplest\n        way to put the Linears and start_embedding in the computation graph\n        '
        start = self.start_embedding
        if self.use_position:
            position = self.position_encoding([0]).squeeze(0)
            start = start + position
        key = self.w_key(start).unsqueeze(0)
        query = self.w_query(start)
        value = self.w_value(start).unsqueeze(0)
        output = self.attention(key.unsqueeze(0), query.unsqueeze(0), value.unsqueeze(0)).squeeze(0)
        return TreeStack(value=Node(initial_value, key, value, output), parent=None, length=1)

    def push_states(self, stacks, values, inputs):
        if False:
            i = 10
            return i + 15
        '\n        Push new inputs to the stacks and rerun attention on them\n\n        Where B is the number of items stacked together, I is input_size\n        stacks: B TreeStacks such as produced by initial_state and/or push_states\n        values: the new items to push on the stacks such as tree nodes or anything\n        inputs: BxI for the new input items\n\n        Runs attention starting from the existing keys & values\n        '
        device = self.w_key.weight.device
        batch_len = len(stacks)
        positions = [x.value.key_stack.shape[0] for x in stacks]
        max_len = max(positions)
        if self.use_position:
            position_encodings = self.position_encoding(positions)
            inputs = inputs + position_encodings
        inputs = self.input_dropout(inputs)
        if len(inputs.shape) == 3:
            if inputs.shape[0] == 1:
                inputs = inputs.squeeze(0)
            else:
                raise ValueError('Expected the inputs to be of shape 1xBxI, got {}'.format(inputs.shape))
        new_keys = self.w_key(inputs)
        key_stack = torch.zeros(batch_len, max_len + 1, self.output_size, device=device)
        key_stack[:, -1, :] = new_keys
        for (stack_idx, stack) in enumerate(stacks):
            key_stack[stack_idx, -positions[stack_idx] - 1:-1, :] = stack.value.key_stack
        new_values = self.w_value(inputs)
        value_stack = torch.zeros(batch_len, max_len + 1, self.output_size, device=device)
        value_stack[:, -1, :] = new_values
        for (stack_idx, stack) in enumerate(stacks):
            value_stack[stack_idx, -positions[stack_idx] - 1:-1, :] = stack.value.value_stack
        query = self.w_query(inputs)
        mask = torch.zeros(batch_len, max_len + 1, device=device, dtype=torch.bool)
        for (stack_idx, stack) in enumerate(stacks):
            if len(stack) < max_len:
                masked = max_len - positions[stack_idx]
                mask[stack_idx, :masked] = True
        batched_output = self.attention(key_stack, query, value_stack, mask)
        new_stacks = []
        for (stack_idx, (stack, node_value, new_key, new_value, output)) in enumerate(zip(stacks, values, key_stack, value_stack, batched_output)):
            new_key_stack = new_key[max_len - positions[stack_idx]:, :]
            new_value_stack = new_value[max_len - positions[stack_idx]:, :]
            if self.length_limit is not None and new_key_stack.shape[0] > self.length_limit + 1:
                new_key_stack = torch.cat([new_key_stack[:1, :], new_key_stack[2:, :]], axis=0)
                new_value_stack = torch.cat([new_value_stack[:1, :], new_value_stack[2:, :]], axis=0)
            new_stacks.append(stack.push(value=Node(node_value, new_key_stack, new_value_stack, output)))
        return new_stacks

    def output(self, stack):
        if False:
            for i in range(10):
                print('nop')
        '\n        Return the last layer of the lstm_hx as the output from a stack\n\n        Refactored so that alternate structures have an easy way of getting the output\n        '
        return stack.value.output