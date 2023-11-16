"""A command line interface to a shift reduce constituency parser.

This follows the work of
Recurrent neural network grammars by Dyer et al
In-Order Transition-based Constituent Parsing by Liu & Zhang

The general outline is:

  Train a model by taking a list of trees, converting them to
    transition sequences, and learning a model which can predict the
    next transition given a current state
  Then, at inference time, repeatedly predict the next transition until parsing is complete

The "transitions" are variations on shift/reduce as per an
intro-to-compilers class.  The idea is that you can treat all of the
words in a sentence as a buffer of tokens, then either "shift" them to
represent a new constituent, or "reduce" one or more constituents to
form a new constituent.

In order to make the runtime a more competitive speed, effort is taken
to batch the transitions and apply multiple transitions at once.  At
train time, batches are groups together by length, and at inference
time, new trees are added to the batch as previous trees on the batch
finish their inference.

There are a few minor differences in the model:
  - The word input is a bi-lstm, not a uni-lstm.
    This gave a small increase in accuracy.
  - The combination of several constituents into one constituent is done
    via a single bi-lstm rather than two separate lstms.  This increases
    speed without a noticeable effect on accuracy.
  - In fact, an even better (in terms of final model accuracy) method
    is to combine the constituents with torch.max, believe it or not
    See lstm_model.py for more details
  - Initializing the embeddings with smaller values than pytorch default
    For example, on a ja_alt dataset, scores went from 0.8980 to 0.8985
    at 200 iterations averaged over 5 trials
  - Training with AdaDelta first, then AdamW or madgrad later improves
    results quite a bit.  See --multistage

A couple experiments which have been tried with little noticeable impact:
  - Combining constituents using the method in the paper (only a trained
    vector at the start instead of both ends) did not affect results
    and is a little slower
  - Using multiple layers of LSTM hidden state for the input to the final
    classification layers didn't help
  - Initializing Linear layers with He initialization and a positive bias
    (to avoid dead connections) had no noticeable effect on accuracy
    0.8396 on it_turin with the original initialization
    0.8401 and 0.8427 on two runs with updated initialization
    (so maybe a small improvement...)
  - Initializing LSTM layers with different gates was slightly worse:
    forget gates of 1.0
    forget gates of 1.0, input gates of -1.0
  - Replacing the LSTMs that make up the Transition and Constituent
    LSTMs with Dynamic Skip LSTMs made no difference, but was slower
  - Highway LSTMs also made no difference
  - Putting labels on the shift transitions (the word or the tag shifted)
    or putting labels on the close transitions didn't help
  - Building larger constituents from the output of the constituent LSTM
    instead of the children constituents hurts scores
    For example, an experiment on ja_alt went from 0.8985 to 0.8964
    when built that way
  - The initial transition scheme implemented was TOP_DOWN.  We tried
    a compound unary option, since this worked so well in the CoreNLP
    constituency parser.  Unfortunately, this is far less effective
    than IN_ORDER.  Both specialized unary matrices and reusing the
    n-ary constituency combination fell short.  On the ja_alt dataset:
      IN_ORDER, max combination method:           0.8985
      TOP_DOWN_UNARY, specialized matrices:       0.8501
      TOP_DOWN_UNARY, max combination method:     0.8508
  - Adding multiple layers of MLP to combine inputs for words made
    no difference in the scores
    Tried both before the LSTM and after
    A simple single layer tensor multiply after the LSTM works well.
    Replacing that with a two layer MLP on the English PTB
    with roberta-base causes a notable drop in scores
    First experiment didn't use the fancy Linear weight init,
    but adding that barely made a difference
      260 training iterations on en_wsj dev, roberta-base
      model as of bb983fd5e912f6706ad484bf819486971742c3d1
      two layer MLP:                    0.9409
      two layer MLP, init weights:      0.9413
      single layer:                     0.9467
  - There is code to rebuild models with a new structure in lstm_model.py
    As part of this, we tried to randomly reinitialize the transitions
    if the transition embedding had gone to 0, which often happens
    This didn't help at all
  - We tried something akin to attention with just the query vector
    over the bert embeddings as a way to mix them, but that did not
    improve scores.
    Example, with a self.bert_layer_mix of size bert_dim x 1:
        mixed_bert_embeddings = []
        for feature in bert_embeddings:
            weighted_feature = self.bert_layer_mix(feature.transpose(1, 2))
            weighted_feature = torch.softmax(weighted_feature, dim=1)
            weighted_feature = torch.matmul(feature, weighted_feature).squeeze(2)
            mixed_bert_embeddings.append(weighted_feature)
        bert_embeddings = mixed_bert_embeddings
    It seems just finetuning the transformer is already enough
    (in general, no need to mix layers at all when finetuning bert embeddings)


The code breakdown is as follows:

  this file: main interface for training or evaluating models
  constituency/trainer.py: contains the training & evaluation code
  constituency/ensemble.py: evaluation code specifically for letting multiple models
    vote on the correct next transition.  a modest improvement.
  constituency/evaluate_treebanks.py: specifically to evaluate multiple parsed treebanks
    against a gold.  in particular, reports whether the theoretical best from those
    parsed treebanks is an improvement (eg, the k-best score as reported by CoreNLP)

  constituency/parse_tree.py: a data structure for representing a parse tree and utility methods
  constituency/tree_reader.py: a module which can read trees from a string or input file

  constituency/tree_stack.py: a linked list which can branch in
    different directions, which will be useful when implementing beam
    search or a dynamic oracle
  constituency/lstm_tree_stack.py: an LSTM over the elements of a TreeStack
  constituency/transformer_tree_stack.py: attempts to run attention over the nodes
    of a tree_stack.  not as effective as the lstm_tree_stack in the initial experiments.
    perhaps it could be refined to work better, though

  constituency/parse_transitions.py: transitions and a State data structure to store them
  constituency/transition_sequence.py: turns ParseTree objects into
    the transition sequences needed to make them

  constituency/base_model.py: operates on the transitions to turn them in to constituents,
    eventually forming one final parse tree composed of all of the constituents
  constituency/lstm_model.py: adds LSTM features to the constituents to predict what the
    correct transition to make is, allowing for predictions on previously unseen text

  constituency/retagging.py: a couple utility methods specifically for retagging
  constituency/utils.py: a couple utility methods

  constituency/dyanmic_oracle.py: a dynamic oracle which currently
    only operates for the inorder transition sequence.
    uses deterministic rules to redo the correct action sequence when
    the parser makes an error.

  constituency/partitioned_transformer.py: implementation of a transformer for self-attention.
     presumably this should help, but we have yet to find a model structure where
     this makes the scores go up.
  constituency/label_attention.py: an even fancier form of transformer based on labeled attention:
     https://arxiv.org/abs/1911.03875
  constituency/positional_encoding.py: so far, just the sinusoidal is here.
     a trained encoding is in partitioned_transformer.py.
     this should probably be refactored to common, especially if used elsewhere.

  stanza/pipeline/constituency_processor.py: interface between this model and the Pipeline

  stanza/utils/datasets/constituency: various scripts and tools for processing constituency datasets

Some alternate optimizer methods:
  adabelief: https://github.com/juntang-zhuang/Adabelief-Optimizer
  madgrad: https://github.com/facebookresearch/madgrad

"""
import argparse
import logging
import os
import re
import torch
import stanza
from stanza.models.common import constant
from stanza.models.common import utils
from stanza.models.constituency import retagging
from stanza.models.constituency import trainer
from stanza.models.constituency.lstm_model import ConstituencyComposition, SentenceBoundary, StackHistory
from stanza.models.constituency.parse_transitions import TransitionScheme
from stanza.models.constituency.utils import DEFAULT_LEARNING_EPS, DEFAULT_LEARNING_RATES, DEFAULT_MOMENTUM, DEFAULT_LEARNING_RHO, DEFAULT_WEIGHT_DECAY, NONLINEARITY, add_predict_output_args, postprocess_predict_output_args
from stanza.resources.common import DEFAULT_MODEL_DIR
logger = logging.getLogger('stanza')

def build_argparse():
    if False:
        print('Hello World!')
    '\n    Adds the arguments for building the con parser\n\n    For the most part, defaults are set to cross-validated values, at least for WSJ\n    '
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/constituency', help='Directory of constituency data.')
    parser.add_argument('--wordvec_dir', type=str, default='extern_data/wordvec', help='Directory of word vectors')
    parser.add_argument('--wordvec_file', type=str, default='', help='File that contains word vectors')
    parser.add_argument('--wordvec_pretrain_file', type=str, default=None, help='Exact name of the pretrain file to read')
    parser.add_argument('--pretrain_max_vocab', type=int, default=250000)
    parser.add_argument('--charlm_forward_file', type=str, default=None, help='Exact path to use for forward charlm')
    parser.add_argument('--charlm_backward_file', type=str, default=None, help='Exact path to use for backward charlm')
    parser.add_argument('--bert_model', type=str, default=None, help='Use an external bert model (requires the transformers package)')
    parser.add_argument('--no_bert_model', dest='bert_model', action='store_const', const=None, help="Don't use bert")
    parser.add_argument('--bert_hidden_layers', type=int, default=4, help='How many layers of hidden state to use from the transformer')
    parser.add_argument('--bert_hidden_layers_original', action='store_const', const=None, dest='bert_hidden_layers', help='Use layers 2,3,4 of the Bert embedding')
    parser.add_argument('--bert_finetune', default=False, action='store_true', help='Finetune the bert (or other transformer)')
    parser.add_argument('--no_bert_finetune', dest='bert_finetune', action='store_false', help="Don't finetune the bert (or other transformer)")
    parser.add_argument('--bert_finetune_layers', default=None, type=int, help='Only finetune this many layers from the transformer')
    parser.add_argument('--bert_finetune_begin_epoch', default=None, type=int, help='Which epoch to start finetuning the transformer')
    parser.add_argument('--bert_finetune_end_epoch', default=None, type=int, help='Which epoch to stop finetuning the transformer')
    parser.add_argument('--bert_learning_rate', default=0.009, type=float, help='Scale the learning rate for transformer finetuning by this much')
    parser.add_argument('--stage1_bert_learning_rate', default=None, type=float, help='Scale the learning rate for transformer finetuning by this much only during an AdaDelta warmup')
    parser.add_argument('--bert_weight_decay', default=0.0001, type=float, help='Scale the weight decay for transformer finetuning by this much')
    parser.add_argument('--stage1_bert_finetune', default=None, action='store_true', help="Finetune the bert (or other transformer) during an AdaDelta warmup, even if the second half doesn't use bert_finetune")
    parser.add_argument('--no_stage1_bert_finetune', dest='stage1_bert_finetune', action='store_false', help="Don't finetune the bert (or other transformer) during an AdaDelta warmup, even if the second half doesn't use bert_finetune")
    parser.add_argument('--tag_embedding_dim', type=int, default=20, help='Embedding size for a tag.  0 turns off the feature')
    parser.add_argument('--delta_embedding_dim', type=int, default=100, help='Embedding size for a delta embedding')
    parser.add_argument('--train_file', type=str, default=None, help='Input file for data loader.')
    parser.add_argument('--silver_file', type=str, default=None, help='Secondary training file.')
    parser.add_argument('--silver_remove_duplicates', default=False, action='store_true', help="Do/don't remove duplicates from the silver training file.  Could be useful for intentionally reweighting some trees")
    parser.add_argument('--eval_file', type=str, default=None, help='Input file for data loader.')
    parser.add_argument('--tokenized_file', type=str, default=None, help='Input file of tokenized text for parsing with parse_text.')
    parser.add_argument('--tokenized_dir', type=str, default=None, help='Input directory of tokenized text for parsing with parse_text.')
    parser.add_argument('--mode', default='train', choices=['train', 'parse_text', 'predict', 'remove_optimizer'])
    parser.add_argument('--num_generate', type=int, default=0, help='When running a dev set, how many sentences to generate beyond the greedy one')
    add_predict_output_args(parser)
    parser.add_argument('--lang', type=str, help='Language')
    parser.add_argument('--shorthand', type=str, help='Treebank shorthand')
    parser.add_argument('--transition_embedding_dim', type=int, default=20, help='Embedding size for a transition')
    parser.add_argument('--transition_hidden_size', type=int, default=20, help='Embedding size for transition stack')
    parser.add_argument('--transition_stack', default=StackHistory.LSTM, type=lambda x: StackHistory[x.upper()], help='How to track transitions over a parse.  {}'.format(', '.join((x.name for x in StackHistory))))
    parser.add_argument('--transition_heads', default=4, type=int, help='How many heads to use in MHA *if* the transition_stack is Attention')
    parser.add_argument('--constituent_stack', default=StackHistory.LSTM, type=lambda x: StackHistory[x.upper()], help='How to track transitions over a parse.  {}'.format(', '.join((x.name for x in StackHistory))))
    parser.add_argument('--constituent_heads', default=8, type=int, help='How many heads to use in MHA *if* the transition_stack is Attention')
    parser.add_argument('--hidden_size', type=int, default=512, help='Size of the output layers for constituency stack and word queue')
    parser.add_argument('--epochs', type=int, default=400)
    parser.add_argument('--epoch_size', type=int, default=5000, help="Runs this many trees in an 'epoch' instead of going through the training dataset exactly once.  Set to 0 to do the whole training set")
    parser.add_argument('--silver_epoch_size', type=int, default=None, help="Runs this many trees in a silver 'epoch'.  If not set, will match --epoch_size")
    parser.add_argument('--multistage', default=True, action='store_true', help='1/2 epochs with adadelta no pattn or lattn, 1/4 with chosen optim and no lattn, 1/4 full model')
    parser.add_argument('--no_multistage', dest='multistage', action='store_false', help="don't do the multistage learning")
    parser.add_argument('--oracle_initial_epoch', type=int, default=1, help='Epoch where we start using the dynamic oracle to let the parser keep going with wrong decisions')
    parser.add_argument('--oracle_frequency', type=float, default=0.8, help='How often to use the oracle vs how often to force the correct transition')
    parser.add_argument('--oracle_forced_errors', type=float, default=0.001, help='Occasionally have the model randomly walk through the state space to try to learn how to recover')
    parser.add_argument('--oracle_level', type=int, default=None, help='Restrict oracle transitions to this level or lower.  0 means off.  None means use all oracle transitions.')
    parser.add_argument('--train_batch_size', type=int, default=30, help='How many trees to train before taking an optimizer step')
    parser.add_argument('--eval_batch_size', type=int, default=50, help='How many trees to batch when running eval')
    parser.add_argument('--save_dir', type=str, default='saved_models/constituency', help='Root dir for saving models.')
    parser.add_argument('--save_name', type=str, default='{shorthand}_{embedding}_{finetune}_constituency.pt', help='File name to save the model')
    parser.add_argument('--save_each_name', type=str, default=None, help='Save each model in sequence to this pattern.  Mostly for testing')
    parser.add_argument('--seed', type=int, default=1234)
    utils.add_device_args(parser)
    parser.add_argument('--learning_rate', default=None, type=float, help='Learning rate for the optimizer.  Reasonable values are 1.0 for adadelta or 0.001 for SGD.  None uses a default for the given optimizer: {}'.format(DEFAULT_LEARNING_RATES))
    parser.add_argument('--learning_eps', default=None, type=float, help='eps value to use in the optimizer.  None uses a default for the given optimizer: {}'.format(DEFAULT_LEARNING_EPS))
    parser.add_argument('--learning_momentum', default=None, type=float, help='Momentum.  None uses a default for the given optimizer: {}'.format(DEFAULT_MOMENTUM))
    parser.add_argument('--learning_weight_decay', default=None, type=float, help='Weight decay (eg, l2 reg) to use in the optimizer')
    parser.add_argument('--learning_rho', default=DEFAULT_LEARNING_RHO, type=float, help='Rho parameter in Adadelta')
    parser.add_argument('--learning_beta2', default=0.999, type=float, help='Beta2 argument for AdamW')
    parser.add_argument('--optim', default=None, help='Optimizer type: SGD, AdamW, Adadelta, AdaBelief, Madgrad')
    parser.add_argument('--stage1_learning_rate', default=None, type=float, help='Learning rate to use in the first stage of --multistage.  None means use default: {}'.format(DEFAULT_LEARNING_RATES['adadelta']))
    parser.add_argument('--learning_rate_warmup', default=0, type=int, help="Number of epochs to ramp up learning rate from 0 to full.  Set to 0 to always use the chosen learning rate.  Currently not functional, as it didn't do anything")
    parser.add_argument('--learning_rate_factor', default=0.6, type=float, help='Plateau learning rate decreate when plateaued')
    parser.add_argument('--learning_rate_patience', default=5, type=int, help='Plateau learning rate patience')
    parser.add_argument('--learning_rate_cooldown', default=10, type=int, help='Plateau learning rate cooldown')
    parser.add_argument('--learning_rate_min_lr', default=None, type=float, help='Plateau learning rate minimum')
    parser.add_argument('--stage1_learning_rate_min_lr', default=None, type=float, help='Plateau learning rate minimum (stage 1)')
    parser.add_argument('--grad_clipping', default=None, type=float, help='Clip abs(grad) to this amount.  Use --no_grad_clipping to turn off grad clipping')
    parser.add_argument('--no_grad_clipping', action='store_const', const=None, dest='grad_clipping', help='Use --no_grad_clipping to turn off grad clipping')
    parser.add_argument('--loss', default='cross', help='cross, large_margin, or focal.  Focal requires `pip install focal_loss_torch`')
    parser.add_argument('--loss_focal_gamma', default=2, type=float, help='gamma value for a focal loss')
    parser.add_argument('--word_dropout', default=0.2, type=float, help='Dropout on the word embedding')
    parser.add_argument('--predict_dropout', default=0.2, type=float, help='Dropout on the final prediction layer')
    parser.add_argument('--lstm_layer_dropout', default=0.0, type=float, help='Dropout in the LSTM layers')
    parser.add_argument('--lstm_input_dropout', default=0.2, type=float, help='Dropout on the input to an LSTM')
    parser.add_argument('--transition_scheme', default=TransitionScheme.IN_ORDER, type=lambda x: TransitionScheme[x.upper()], help='Transition scheme to use.  {}'.format(', '.join((x.name for x in TransitionScheme))))
    parser.add_argument('--reversed', default=False, action='store_true', help='Do the transition sequence reversed')
    parser.add_argument('--combined_dummy_embedding', default=True, action='store_true', help='Use the same embedding for dummy nodes and the vectors used when combining constituents')
    parser.add_argument('--no_combined_dummy_embedding', dest='combined_dummy_embedding', action='store_false', help="Don't use the same embedding for dummy nodes and the vectors used when combining constituents")
    parser.add_argument('--nonlinearity', default='relu', choices=NONLINEARITY.keys(), help='Nonlinearity to use in the model.  relu is a noticeable improvement over tanh')
    parser.add_argument('--maxout_k', default=None, type=int, help='Use maxout layers instead of a nonlinearity for the output layers')
    parser.add_argument('--use_silver_words', default=True, dest='use_silver_words', action='store_true', help="Train/don't train word vectors for words only in the silver dataset")
    parser.add_argument('--no_use_silver_words', default=True, dest='use_silver_words', action='store_false', help="Train/don't train word vectors for words only in the silver dataset")
    parser.add_argument('--rare_word_unknown_frequency', default=0.02, type=float, help='How often to replace a rare word with UNK when training')
    parser.add_argument('--rare_word_threshold', default=0.02, type=float, help='How many words to consider as rare words as a fraction of the dataset')
    parser.add_argument('--tag_unknown_frequency', default=0.001, type=float, help='How often to replace a tag with UNK when training')
    parser.add_argument('--num_lstm_layers', default=2, type=int, help='How many layers to use in the LSTMs')
    parser.add_argument('--num_tree_lstm_layers', default=None, type=int, help='How many layers to use in the TREE_LSTMs, if used.  This also increases the width of the word outputs to match the tree lstm inputs.  Default 2 if TREE_LSTM or TREE_LSTM_CX, 1 otherwise')
    parser.add_argument('--num_output_layers', default=3, type=int, help='How many layers to use at the prediction level')
    parser.add_argument('--sentence_boundary_vectors', default=SentenceBoundary.EVERYTHING, type=lambda x: SentenceBoundary[x.upper()], help='Vectors to learn at the start & end of sentences.  {}'.format(', '.join((x.name for x in SentenceBoundary))))
    parser.add_argument('--constituency_composition', default=ConstituencyComposition.MAX, type=lambda x: ConstituencyComposition[x.upper()], help='How to build a new composition from its children.  {}'.format(', '.join((x.name for x in ConstituencyComposition))))
    parser.add_argument('--reduce_heads', default=8, type=int, help='Number of attn heads to use when reducing children into a parent tree (constituency_composition == attn)')
    parser.add_argument('--reduce_position', default=None, type=int, help="Dimension of position vector to use when reducing children.  None means 1/4 hidden_size, 0 means don't use (constituency_composition == key | untied_key)")
    parser.add_argument('--relearn_structure', action='store_true', help='Starting from an existing checkpoint, add or remove pattn / lattn.  One thing that works well is to train an initial model using adadelta with no pattn, then add pattn with adamw')
    parser.add_argument('--finetune', action='store_true', help='Load existing model during `train` mode from `load_name` path')
    parser.add_argument('--checkpoint_save_name', type=str, default=None, help='File name to save the most recent checkpoint')
    parser.add_argument('--no_checkpoint', dest='checkpoint', action='store_false', help="Don't save checkpoints")
    parser.add_argument('--load_name', type=str, default=None, help='Model to load when finetuning, evaluating, or manipulating an existing file')
    parser.add_argument('--load_package', type=str, default=None, help='Download an existing stanza package & use this for tests, finetuning, etc')
    retagging.add_retag_args(parser)
    parser.add_argument('--pattn_d_model', default=1024, type=int, help='Partitioned attention model dimensionality')
    parser.add_argument('--pattn_morpho_emb_dropout', default=0.2, type=float, help='Dropout rate for morphological features obtained from pretrained model')
    parser.add_argument('--pattn_encoder_max_len', default=512, type=int, help='Max length that can be put into the transformer attention layer')
    parser.add_argument('--pattn_num_heads', default=8, type=int, help='Partitioned attention model number of attention heads')
    parser.add_argument('--pattn_d_kv', default=64, type=int, help='Size of the query and key vector')
    parser.add_argument('--pattn_d_ff', default=2048, type=int, help='Size of the intermediate vectors in the feed-forward sublayer')
    parser.add_argument('--pattn_relu_dropout', default=0.1, type=float, help='ReLU dropout probability in feed-forward sublayer')
    parser.add_argument('--pattn_residual_dropout', default=0.2, type=float, help='Residual dropout probability for all residual connections')
    parser.add_argument('--pattn_attention_dropout', default=0.2, type=float, help='Attention dropout probability')
    parser.add_argument('--pattn_num_layers', default=0, type=int, help='Number of layers for the Partitioned Attention.  Currently turned off')
    parser.add_argument('--pattn_bias', default=False, action='store_true', help='Whether or not to learn an additive bias')
    parser.add_argument('--pattn_timing', default='sin', choices=['learned', 'sin'], help='Use a learned embedding or a sin embedding')
    parser.add_argument('--lattn_d_input_proj', default=None, type=int, help='If set, project the non-positional inputs down to this size before proceeding.')
    parser.add_argument('--lattn_d_kv', default=64, type=int, help='Dimension of the key/query vector')
    parser.add_argument('--lattn_d_proj', default=64, type=int, help='Dimension of the output vector from each label attention head')
    parser.add_argument('--lattn_resdrop', default=True, action='store_true', help='Whether or not to use Residual Dropout')
    parser.add_argument('--lattn_pwff', default=True, action='store_true', help='Whether or not to use a Position-wise Feed-forward Layer')
    parser.add_argument('--lattn_q_as_matrix', default=False, action='store_true', help='Whether or not Label Attention uses learned query vectors. False means it does')
    parser.add_argument('--lattn_partitioned', default=True, action='store_true', help='Whether or not it is partitioned')
    parser.add_argument('--no_lattn_partitioned', default=True, action='store_false', dest='lattn_partitioned', help='Whether or not it is partitioned')
    parser.add_argument('--lattn_combine_as_self', default=False, action='store_true', help='Whether or not the layer uses concatenation. False means it does')
    parser.add_argument('--lattn_d_l', default=32, type=int, help='Number of labels')
    parser.add_argument('--lattn_attention_dropout', default=0.2, type=float, help='Dropout for attention layer')
    parser.add_argument('--lattn_d_ff', default=2048, type=int, help='Dimension of the Feed-forward layer')
    parser.add_argument('--lattn_relu_dropout', default=0.2, type=float, help='Relu dropout for the label attention')
    parser.add_argument('--lattn_residual_dropout', default=0.2, type=float, help='Residual dropout for the label attention')
    parser.add_argument('--lattn_combined_input', default=True, action='store_true', help='Combine all inputs for the lattn, not just the pattn')
    parser.add_argument('--use_lattn', default=False, action='store_true', help='Use the lattn layers - currently turned off')
    parser.add_argument('--no_lattn_combined_input', dest='lattn_combined_input', action='store_false', help="Don't combine all inputs for the lattn, not just the pattn")
    parser.add_argument('--log_norms', default=False, action='store_true', help='Log the parameters norms while training.  A very noisy option')
    parser.add_argument('--log_shapes', default=False, action='store_true', help='Log the parameters shapes at the beginning')
    parser.add_argument('--watch_regex', default=None, help='regex to describe which weights and biases to output, if any')
    parser.add_argument('--wandb', action='store_true', help='Start a wandb session and write the results of training.  Only applies to training.  Use --wandb_name instead to specify a name')
    parser.add_argument('--wandb_name', default=None, help='Name of a wandb session to start when training.  Will default to the dataset short name')
    parser.add_argument('--wandb_norm_regex', default=None, help='Log on wandb any tensor whose norm matches this matrix.  Might get cluttered?')
    return parser

def build_model_filename(args):
    if False:
        i = 10
        return i + 15
    embedding = utils.embedding_name(args)
    maybe_finetune = 'finetuned' if args['bert_finetune'] or args['stage1_bert_finetune'] else ''
    transformer_finetune_begin = '%d' % args['bert_finetune_begin_epoch'] if args['bert_finetune_begin_epoch'] is not None else ''
    model_save_file = args['save_name'].format(shorthand=args['shorthand'], embedding=embedding, finetune=maybe_finetune, transformer_finetune_begin=transformer_finetune_begin, transition_scheme=args['transition_scheme'].name.lower().replace('_', ''), trans_layers=args['bert_hidden_layers'], seed=args['seed'])
    model_save_file = re.sub('_+', '_', model_save_file)
    logger.info('Expanded save_name: %s', model_save_file)
    model_dir = os.path.split(model_save_file)[0]
    if model_dir != args['save_dir']:
        model_save_file = os.path.join(args['save_dir'], model_save_file)
    return model_save_file

def parse_args(args=None):
    if False:
        i = 10
        return i + 15
    parser = build_argparse()
    args = parser.parse_args(args=args)
    if not args.lang and args.shorthand and (len(args.shorthand.split('_', maxsplit=1)) == 2):
        args.lang = args.shorthand.split('_')[0]
    if args.stage1_bert_learning_rate is None:
        args.stage1_bert_learning_rate = args.bert_learning_rate
    if args.optim is None and args.mode == 'train':
        if not args.multistage:
            args.optim = 'adadelta'
        elif args.bert_finetune or args.stage1_bert_finetune:
            logger.info('Multistage training is set, optimizer is not chosen, and bert finetuning is active.  Will use AdamW as the second stage optimizer.')
            args.optim = 'adamw'
        else:
            try:
                import madgrad
                args.optim = 'madgrad'
                logger.info('Multistage training is set, optimizer is not chosen, and MADGRAD is available.  Will use MADGRAD as the second stage optimizer.')
            except ModuleNotFoundError as e:
                logger.warning('Multistage training is set.  Best models are with MADGRAD, but it is not installed.  Will use AdamW for the second stage optimizer.  Consider installing MADGRAD')
                args.optim = 'adamw'
    if args.mode == 'train':
        if args.learning_rate is None:
            args.learning_rate = DEFAULT_LEARNING_RATES.get(args.optim.lower(), None)
        if args.learning_eps is None:
            args.learning_eps = DEFAULT_LEARNING_EPS.get(args.optim.lower(), None)
        if args.learning_momentum is None:
            args.learning_momentum = DEFAULT_MOMENTUM.get(args.optim.lower(), None)
        if args.learning_weight_decay is None:
            args.learning_weight_decay = DEFAULT_WEIGHT_DECAY.get(args.optim.lower(), None)
        if args.stage1_learning_rate is None:
            args.stage1_learning_rate = DEFAULT_LEARNING_RATES['adadelta']
        if args.stage1_bert_finetune is None:
            args.stage1_bert_finetune = args.bert_finetune
        if args.learning_rate_min_lr is None:
            args.learning_rate_min_lr = args.learning_rate * 0.02
        if args.stage1_learning_rate_min_lr is None:
            args.stage1_learning_rate_min_lr = args.stage1_learning_rate * 0.02
    if args.reduce_position is None:
        args.reduce_position = args.hidden_size // 4
    if args.num_tree_lstm_layers is None:
        if args.constituency_composition in (ConstituencyComposition.TREE_LSTM, ConstituencyComposition.TREE_LSTM_CX):
            args.num_tree_lstm_layers = 2
        else:
            args.num_tree_lstm_layers = 1
    if args.wandb_name or args.wandb_norm_regex:
        args.wandb = True
    args = vars(args)
    retagging.postprocess_args(args)
    postprocess_predict_output_args(args)
    model_save_file = build_model_filename(args)
    args['save_name'] = model_save_file
    if args['checkpoint']:
        args['checkpoint_save_name'] = utils.checkpoint_name(args['save_dir'], model_save_file, args['checkpoint_save_name'])
    return args

def main(args=None):
    if False:
        print('Hello World!')
    '\n    Main function for building con parser\n\n    Processes args, calls the appropriate function for the chosen --mode\n    '
    args = parse_args(args=args)
    utils.set_random_seed(args['seed'])
    logger.info('Running constituency parser in %s mode', args['mode'])
    logger.debug('Using device: %s', args['device'])
    model_save_each_file = None
    if args['save_each_name']:
        model_save_each_file = os.path.join(args['save_dir'], args['save_each_name'])
        try:
            model_save_each_file % 1
        except TypeError:
            pieces = os.path.splitext(model_save_each_file)
            model_save_each_file = pieces[0] + '_%04d' + pieces[1]
    model_load_file = args['save_name']
    if args['load_name']:
        if os.path.exists(args['load_name']):
            model_load_file = args['load_name']
        else:
            model_load_file = os.path.join(args['save_dir'], args['load_name'])
    elif args['load_package']:
        if args['lang'] is None:
            lang_pieces = args['load_package'].split('_', maxsplit=1)
            try:
                lang = constant.lang_to_langcode(lang_pieces[0])
            except ValueError as e:
                raise ValueError('--lang not specified, and the start of the --load_package name, %s, is not a known language.  Please check the values of those parameters' % args['load_package']) from e
            args['lang'] = lang
            args['load_package'] = lang_pieces[1]
        stanza.download(args['lang'], processors='constituency', package={'constituency': args['load_package']})
        model_load_file = os.path.join(DEFAULT_MODEL_DIR, args['lang'], 'constituency', args['load_package'] + '.pt')
        if not os.path.exists(model_load_file):
            raise FileNotFoundError("Expected the downloaded model file for language %s package %s to be in %s, but there is nothing there.  Perhaps the package name doesn't exist?" % (args['lang'], args['load_package'], model_load_file))
        else:
            logger.info('Model for language %s package %s is in %s', args['lang'], args['load_package'], model_load_file)
    retag_pipeline = retagging.build_retag_pipeline(args)
    if args['mode'] == 'train':
        trainer.train(args, model_load_file, model_save_each_file, retag_pipeline)
    elif args['mode'] == 'predict':
        trainer.evaluate(args, model_load_file, retag_pipeline)
    elif args['mode'] == 'parse_text':
        trainer.load_model_parse_text(args, model_load_file, retag_pipeline)
    elif args['mode'] == 'remove_optimizer':
        trainer.remove_optimizer(args, args['save_name'], model_load_file)
if __name__ == '__main__':
    main()