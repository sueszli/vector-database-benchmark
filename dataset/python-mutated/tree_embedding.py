"""
A module to use a Constituency Parser to make an embedding for a tree

The embedding can be produced just from the words and the top of the
tree, or it can be done with a form of attention over the nodes

Can be done over an existing parse tree or unparsed text
"""
import torch
import torch.nn as nn
from stanza.models.constituency.trainer import Trainer

class TreeEmbedding(nn.Module):

    def __init__(self, constituency_parser, args):
        if False:
            return 10
        super(TreeEmbedding, self).__init__()
        self.config = {'all_words': args['all_words'], 'backprop': args['backprop'], 'node_attn': args['node_attn'], 'top_layer': args['top_layer']}
        self.constituency_parser = constituency_parser
        self.hidden_size = self.constituency_parser.hidden_size + self.constituency_parser.transition_hidden_size
        if self.config['all_words']:
            self.hidden_size += self.constituency_parser.hidden_size * self.constituency_parser.num_tree_lstm_layers
        else:
            self.hidden_size += self.constituency_parser.hidden_size * self.constituency_parser.num_tree_lstm_layers * 2
        if self.config['node_attn']:
            self.query = nn.Linear(self.constituency_parser.hidden_size, self.constituency_parser.hidden_size)
            self.key = nn.Linear(self.hidden_size, self.constituency_parser.hidden_size)
            self.value = nn.Linear(self.constituency_parser.hidden_size, self.constituency_parser.hidden_size)
            self.output_size = self.constituency_parser.hidden_size * self.constituency_parser.num_tree_lstm_layers
        else:
            self.output_size = self.hidden_size

    def embed_trees(self, inputs):
        if False:
            while True:
                i = 10
        if self.config['backprop']:
            states = self.constituency_parser.analyze_trees(inputs)
        else:
            with torch.no_grad():
                states = self.constituency_parser.analyze_trees(inputs)
        constituent_lists = [x.constituents for x in states]
        states = [x.state for x in states]
        word_begin_hx = torch.stack([state.word_queue[0].hx for state in states])
        word_end_hx = torch.stack([state.word_queue[state.word_position].hx for state in states])
        transition_hx = torch.stack([self.constituency_parser.transition_stack.output(state.transitions) for state in states])
        if self.config['top_layer']:
            constituent_hx = torch.stack([self.constituency_parser.constituent_stack.output(state.constituents) for state in states])
        else:
            constituent_hx = torch.cat([constituents[-2].tree_hx for constituents in constituent_lists], dim=0)
        if self.config['all_words']:
            key = [torch.stack([torch.cat([word.hx, thx, chx]) for word in state.word_queue], dim=0) for (state, thx, chx) in zip(states, transition_hx, constituent_hx)]
        else:
            key = torch.cat((word_begin_hx, word_end_hx, transition_hx, constituent_hx), dim=1).unsqueeze(1)
        if not self.config['node_attn']:
            return key
        key = [self.key(x) for x in key]
        node_hx = [torch.stack([con.tree_hx for con in constituents], dim=0) for constituents in constituent_lists]
        queries = [self.query(nhx).reshape(nhx.shape[0], -1) for nhx in node_hx]
        values = [self.value(nhx).reshape(nhx.shape[0], -1) for nhx in node_hx]
        attn = [torch.matmul(q, k.transpose(0, 1)) for (q, k) in zip(queries, key)]
        attn = [torch.softmax(x, dim=0) for x in attn]
        previous_layer = [torch.matmul(weight.transpose(0, 1), value) for (weight, value) in zip(attn, values)]
        return previous_layer

    def forward(self, inputs):
        if False:
            for i in range(10):
                print('nop')
        return embed_trees(self, inputs)

    def get_norms(self):
        if False:
            while True:
                i = 10
        lines = ['constituency_parser.' + x for x in self.constituency_parser.get_norms()]
        for (name, param) in self.named_parameters():
            if param.requires_grad and (not name.startswith('constituency_parser.')):
                lines.append('%s %.6g' % (name, torch.norm(param).item()))
        return lines

    def get_params(self, skip_modules=True):
        if False:
            while True:
                i = 10
        model_state = self.state_dict()
        skipped = [k for k in model_state.keys() if k.startswith('constituency_parser.')]
        for k in skipped:
            del model_state[k]
        parser = self.constituency_parser.get_params(skip_modules)
        params = {'model': model_state, 'constituency': parser, 'config': self.config}
        return params

    @staticmethod
    def from_parser_file(args, foundation_cache=None):
        if False:
            print('Hello World!')
        constituency_parser = Trainer.load(args['model'], args, foundation_cache)
        return TreeEmbedding(constituency_parser.model, args)

    @staticmethod
    def model_from_params(params, args, foundation_cache=None):
        if False:
            for i in range(10):
                print('nop')
        constituency_parser = Trainer.model_from_params(params['constituency'], args, foundation_cache)
        model = TreeEmbedding(constituency_parser, params['config'])
        model.load_state_dict(params['model'], strict=False)
        return model