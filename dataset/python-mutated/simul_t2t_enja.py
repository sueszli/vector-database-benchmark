import os
from fairseq import checkpoint_utils, tasks
import sentencepiece as spm
import torch
try:
    from simuleval import READ_ACTION, WRITE_ACTION, DEFAULT_EOS
    from simuleval.agents import TextAgent
except ImportError:
    print("Please install simuleval 'pip install simuleval'")
BOS_PREFIX = '▁'

class SimulTransTextAgentJA(TextAgent):
    """
    Simultaneous Translation
    Text agent for Japanese
    """

    def __init__(self, args):
        if False:
            while True:
                i = 10
        self.gpu = getattr(args, 'gpu', False)
        self.max_len = args.max_len
        self.load_model_vocab(args)
        self.build_word_splitter(args)
        self.eos = DEFAULT_EOS

    def initialize_states(self, states):
        if False:
            while True:
                i = 10
        states.incremental_states = dict()
        states.incremental_states['online'] = dict()

    def to_device(self, tensor):
        if False:
            for i in range(10):
                print('nop')
        if self.gpu:
            return tensor.cuda()
        else:
            return tensor.cpu()

    def load_model_vocab(self, args):
        if False:
            while True:
                i = 10
        filename = args.model_path
        if not os.path.exists(filename):
            raise IOError('Model file not found: {}'.format(filename))
        state = checkpoint_utils.load_checkpoint_to_cpu(filename)
        task_args = state['cfg']['task']
        task_args.data = args.data_bin
        task = tasks.setup_task(task_args)
        state['cfg']['model'].load_pretrained_encoder_from = None
        state['cfg']['model'].load_pretrained_decoder_from = None
        self.model = task.build_model(state['cfg']['model'])
        self.model.load_state_dict(state['model'], strict=True)
        self.model.eval()
        self.model.share_memory()
        if self.gpu:
            self.model.cuda()
        self.dict = {}
        self.dict['tgt'] = task.target_dictionary
        self.dict['src'] = task.source_dictionary

    @staticmethod
    def add_args(parser):
        if False:
            return 10
        parser.add_argument('--model-path', type=str, required=True, help='path to your pretrained model.')
        parser.add_argument('--data-bin', type=str, required=True, help='Path of data binary')
        parser.add_argument('--max-len', type=int, default=100, help='Max length of translation')
        parser.add_argument('--tgt-splitter-type', type=str, default='SentencePiece', help='Subword splitter type for target text.')
        parser.add_argument('--tgt-splitter-path', type=str, default=None, help='Subword splitter model path for target text.')
        parser.add_argument('--src-splitter-type', type=str, default='SentencePiece', help='Subword splitter type for source text.')
        parser.add_argument('--src-splitter-path', type=str, default=None, help='Subword splitter model path for source text.')
        return parser

    def build_word_splitter(self, args):
        if False:
            while True:
                i = 10
        self.spm = {}
        for lang in ['src', 'tgt']:
            if getattr(args, f'{lang}_splitter_type', None):
                path = getattr(args, f'{lang}_splitter_path', None)
                if path:
                    self.spm[lang] = spm.SentencePieceProcessor()
                    self.spm[lang].Load(path)

    def segment_to_units(self, segment, states):
        if False:
            for i in range(10):
                print('nop')
        return self.spm['src'].EncodeAsPieces(segment)

    def update_model_encoder(self, states):
        if False:
            while True:
                i = 10
        if len(states.units.source) == 0:
            return
        src_indices = [self.dict['src'].index(x) for x in states.units.source.value]
        if states.finish_read():
            src_indices += [self.dict['tgt'].eos_index]
        src_indices = self.to_device(torch.LongTensor(src_indices).unsqueeze(0))
        src_lengths = self.to_device(torch.LongTensor([src_indices.size(1)]))
        states.encoder_states = self.model.encoder(src_indices, src_lengths)
        torch.cuda.empty_cache()

    def update_states_read(self, states):
        if False:
            i = 10
            return i + 15
        self.update_model_encoder(states)

    def units_to_segment(self, units, states):
        if False:
            return 10
        token = units.value.pop()
        if token == self.dict['tgt'].eos_word or len(states.segments.target) > self.max_len:
            return DEFAULT_EOS
        if BOS_PREFIX == token:
            return None
        if token[0] == BOS_PREFIX:
            return token[1:]
        else:
            return token

    def policy(self, states):
        if False:
            i = 10
            return i + 15
        if not getattr(states, 'encoder_states', None):
            return READ_ACTION
        tgt_indices = self.to_device(torch.LongTensor([self.model.decoder.dictionary.eos()] + [self.dict['tgt'].index(x) for x in states.units.target.value if x is not None]).unsqueeze(0))
        states.incremental_states['steps'] = {'src': states.encoder_states['encoder_out'][0].size(0), 'tgt': 1 + len(states.units.target)}
        states.incremental_states['online']['only'] = torch.BoolTensor([not states.finish_read()])
        (x, outputs) = self.model.decoder.forward(prev_output_tokens=tgt_indices, encoder_out=states.encoder_states, incremental_state=states.incremental_states)
        states.decoder_out = x
        torch.cuda.empty_cache()
        if outputs.action == 0:
            return READ_ACTION
        else:
            return WRITE_ACTION

    def predict(self, states):
        if False:
            print('Hello World!')
        decoder_states = states.decoder_out
        lprobs = self.model.get_normalized_probs([decoder_states[:, -1:]], log_probs=True)
        index = lprobs.argmax(dim=-1)[0, 0].item()
        if index != self.dict['tgt'].eos_index:
            token = self.dict['tgt'].string([index])
        else:
            token = self.dict['tgt'].eos_word
        return token