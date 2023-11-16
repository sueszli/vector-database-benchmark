import torch
from torch import nn
from functorch.dim import dims, dimlists, softmax, cat
import math

class Linear(nn.Linear):

    def forward(self, input):
        if False:
            while True:
                i = 10
        (ci, co) = dims()
        b = dimlists()
        result = (input[b, ci] * self.weight[co, ci]).sum(ci) + self.bias[co]
        return result.order(b, co)

class BertSelfAttention(nn.Module):

    def __init__(self, hidden_size, num_attention_heads, attention_probs_dropout_prob, position_embedding_type=None, max_position_embeddings=None, linear=Linear):
        if False:
            return 10
        super().__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(f'The hidden size ({hidden_size}) is not a multiple of the number of attention heads ({num_attention_heads})')
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.query = linear(hidden_size, self.all_head_size)
        self.key = linear(hidden_size, self.all_head_size)
        self.value = linear(hidden_size, self.all_head_size)
        self.dropout_prob = attention_probs_dropout_prob
        self.position_embedding_type = position_embedding_type
        if self.position_embedding_type is not None:
            assert max_position_embeddings is not None
            self.max_position_embeddings = max_position_embeddings
            self.distance_embedding = nn.Embedding(2 * max_position_embeddings - 1, self.attention_head_size)

    def forward(self, hidden_states, past_key_value=None):
        if False:
            i = 10
            return i + 15
        q = self.query(hidden_states)
        k = self.key(hidden_states)
        v = self.value(hidden_states)
        (batch, query_sequence, key_sequence, heads, features) = dims()
        heads.size = self.num_attention_heads
        q = q[batch, query_sequence, [heads, features]]
        k = k[batch, key_sequence, [heads, features]]
        v = v[batch, key_sequence, [heads, features]]
        if past_key_value is not None:
            extended_key_sequence = dims()
            key_past = past_key_value[0][batch, heads, key_sequence, features]
            value_past = past_key_value[1][batch, heads, key_sequence, features]
            k = cat([key_past, k], key_sequence, extended_key_sequence)
            v = cat([value_past, v], key_sequence, extended_key_sequence)
            key_sequence = extended_key_sequence
        attention_scores = (q * k).sum(features) / math.sqrt(features.size)
        if self.position_embedding_type is not None:
            distance = query_sequence - key_sequence
            assert key_sequence.size <= self.max_position_embeddings
            positional_embedding = self.distance_embedding.weight[self.max_position_embeddings - 1 + distance, features]
            if self.position_embedding_type == 'relative_key':
                relative_position_scores = (q * positional_embedding).sum(features)
                attention_scores = attention_scores + relative_position_scores
            elif self.position_embedding_type == 'relative_key_query':
                relative_position_scores_query = (q * positional_embedding).sum(features)
                relative_position_scores_key = (k * positional_embedding).sum(features)
                attention_scores = attention_scores + relative_position_scores_query + relative_position_scores_key
        attention_probs = attention_scores
        attention_probs = softmax(attention_scores, dim=key_sequence)
        attention_probs = torch.nn.functional.dropout(attention_probs, p=self.dropout_prob)
        context_layer = (attention_probs * v).sum(key_sequence)
        return context_layer.order(batch, query_sequence, [heads, features])