"""
Title: GPT text generation from scratch with KerasNLP
Author: [Jesse Chan](https://github.com/jessechancy)
Date created: 2022/07/25
Last modified: 2022/07/25
Description: Using KerasNLP to train a mini-GPT model for text generation.
Accelerator: GPU
"""
'\n## Introduction\n\nIn this example, we will use KerasNLP to build a scaled down Generative\nPre-Trained (GPT) model. GPT is a Transformer-based model that allows you to generate\nsophisticated text from a prompt.\n\nWe will train the model on the [simplebooks-92](https://arxiv.org/abs/1911.12391) corpus,\nwhich is a dataset made from several novels. It is a good dataset for this example since\nit has a small vocabulary and high word frequency, which is beneficial when training a\nmodel with few parameters.\n\nThis example combines concepts from\n[Text generation with a miniature GPT](https://keras.io/examples/generative/text_generation_with_miniature_gpt/)\nwith KerasNLP abstractions. We will demonstrate how KerasNLP tokenization, layers and\nmetrics simplify the training\nprocess, and then show how to generate output text using the KerasNLP sampling utilities.\n\nNote: If you are running this example on a Colab,\nmake sure to enable GPU runtime for faster training.\n\nThis example requires KerasNLP. You can install it via the following command:\n`pip install keras-nlp`\n'
'\n## Setup\n'
import os
import keras_nlp
import keras
import tensorflow.data as tf_data
import tensorflow.strings as tf_strings
'\n## Settings & hyperparameters\n'
BATCH_SIZE = 64
MIN_STRING_LEN = 512
SEQ_LEN = 128
EMBED_DIM = 256
FEED_FORWARD_DIM = 128
NUM_HEADS = 3
NUM_LAYERS = 2
VOCAB_SIZE = 5000
EPOCHS = 5
NUM_TOKENS_TO_GENERATE = 80
"\n## Load the data\n\nNow, let's download the dataset! The SimpleBooks dataset consists of 1,573 Gutenberg books, and has\none of the smallest vocabulary size to word-level tokens ratio. It has a vocabulary size of ~98k,\na third of WikiText-103's, with around the same number of tokens (~100M). This makes it easy to fit a small model.\n"
keras.utils.get_file(origin='https://dldata-public.s3.us-east-2.amazonaws.com/simplebooks.zip', extract=True)
dir = os.path.expanduser('~/.keras/datasets/simplebooks/')
raw_train_ds = tf_data.TextLineDataset(dir + 'simplebooks-92-raw/train.txt').filter(lambda x: tf_strings.length(x) > MIN_STRING_LEN).batch(BATCH_SIZE).shuffle(buffer_size=256)
raw_val_ds = tf_data.TextLineDataset(dir + 'simplebooks-92-raw/valid.txt').filter(lambda x: tf_strings.length(x) > MIN_STRING_LEN).batch(BATCH_SIZE)
'\n## Train the tokenizer\n\nWe train the tokenizer from the training dataset for a vocabulary size of `VOCAB_SIZE`,\nwhich is a tuned hyperparameter. We want to limit the vocabulary as much as possible, as\nwe will see later on\nthat it has a large effect on the number of model parameters. We also don\'t want to include\n*too few* vocabulary terms, or there would be too many out-of-vocabulary (OOV) sub-words. In\naddition, three tokens are reserved in the vocabulary:\n\n- `"[PAD]"` for padding sequences to `SEQ_LEN`. This token has index 0 in both\n`reserved_tokens` and `vocab`, since `WordPieceTokenizer` (and other layers) consider\n`0`/`vocab[0]` as the default padding.\n- `"[UNK]"` for OOV sub-words, which should match the default `oov_token="[UNK]"` in\n`WordPieceTokenizer`.\n- `"[BOS]"` stands for beginning of sentence, but here technically it is a token\nrepresenting the beginning of each line of training data.\n'
vocab = keras_nlp.tokenizers.compute_word_piece_vocabulary(raw_train_ds, vocabulary_size=VOCAB_SIZE, lowercase=True, reserved_tokens=['[PAD]', '[UNK]', '[BOS]'])
'\n## Load tokenizer\n\nWe use the vocabulary data to initialize\n`keras_nlp.tokenizers.WordPieceTokenizer`. WordPieceTokenizer is an efficient\nimplementation of the WordPiece algorithm used by BERT and other models. It will strip,\nlower-case and do other irreversible preprocessing operations.\n'
tokenizer = keras_nlp.tokenizers.WordPieceTokenizer(vocabulary=vocab, sequence_length=SEQ_LEN, lowercase=True)
'\n## Tokenize data\n\nWe preprocess the dataset by tokenizing and splitting it into `features` and `labels`.\n'
start_packer = keras_nlp.layers.StartEndPacker(sequence_length=SEQ_LEN, start_value=tokenizer.token_to_id('[BOS]'))

def preprocess(inputs):
    if False:
        print('Hello World!')
    outputs = tokenizer(inputs)
    features = start_packer(outputs)
    labels = outputs
    return (features, labels)
train_ds = raw_train_ds.map(preprocess, num_parallel_calls=tf_data.AUTOTUNE).prefetch(tf_data.AUTOTUNE)
val_ds = raw_val_ds.map(preprocess, num_parallel_calls=tf_data.AUTOTUNE).prefetch(tf_data.AUTOTUNE)
'\n## Build the model\n\nWe create our scaled down GPT model with the following layers:\n\n- One `keras_nlp.layers.TokenAndPositionEmbedding` layer, which combines the embedding\nfor the token and its position.\n- Multiple `keras_nlp.layers.TransformerDecoder` layers, with the default causal masking.\nThe layer has no cross-attention when run with decoder sequence only.\n- One final dense linear layer\n'
inputs = keras.layers.Input(shape=(None,), dtype='int32')
embedding_layer = keras_nlp.layers.TokenAndPositionEmbedding(vocabulary_size=VOCAB_SIZE, sequence_length=SEQ_LEN, embedding_dim=EMBED_DIM, mask_zero=True)
x = embedding_layer(inputs)
for _ in range(NUM_LAYERS):
    decoder_layer = keras_nlp.layers.TransformerDecoder(num_heads=NUM_HEADS, intermediate_dim=FEED_FORWARD_DIM)
    x = decoder_layer(x)
outputs = keras.layers.Dense(VOCAB_SIZE)(x)
model = keras.Model(inputs=inputs, outputs=outputs)
loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
perplexity = keras_nlp.metrics.Perplexity(from_logits=True, mask_token_id=0)
model.compile(optimizer='adam', loss=loss_fn, metrics=[perplexity])
"\nLet's take a look at our model summary - a large majority of the\nparameters are in the `token_and_position_embedding` and the output `dense` layer!\nThis means that the vocabulary size (`VOCAB_SIZE`) has a large effect on the size of the model,\nwhile the number of Transformer decoder layers (`NUM_LAYERS`) doesn't affect it as much.\n"
model.summary()
"\n## Training\n\nNow that we have our model, let's train it with the `fit()` method.\n"
model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)
'\n## Inference\n\nWith our trained model, we can test it out to gauge its performance. To do this\nwe can seed our model with an input sequence starting with the `"[BOS]"` token,\nand progressively sample the model by making predictions for each subsequent\ntoken in a loop.\n\nTo start lets build a prompt with the same shape as our model inputs, containing\nonly the `"[BOS]"` token.\n'
prompt_tokens = start_packer(tokenizer(['']))
prompt_tokens
'\nWe will use the `keras_nlp.samplers` module for inference, which requires a\ncallback function wrapping the model we just trained. This wrapper calls\nthe model and returns the logit predictions for the current token we are\ngenerating.\n\nNote: There are two pieces of more advanced functionality available when\ndefining your callback. The first is the ability to take in a `cache` of states\ncomputed in previous generation steps, which can be used to speed up generation.\nThe second is the ability to output the final dense "hidden state" of each\ngenerated token. This is used by `keras_nlp.samplers.ContrastiveSampler`, which\navoids repetition by penalizing repeated hidden states. Both are optional, and\nwe will ignore them for now.\n'

def next(prompt, cache, index):
    if False:
        for i in range(10):
            print('nop')
    logits = model(prompt)[:, index - 1, :]
    hidden_states = None
    return (logits, hidden_states, cache)
"\nCreating the wrapper function is the most complex part of using these functions. Now that\nit's done, let's test out the different utilities, starting with greedy search.\n"
'\n### Greedy search\n\nWe greedily pick the most probable token at each timestep. In other words, we get the\nargmax of the model output.\n'
sampler = keras_nlp.samplers.GreedySampler()
output_tokens = sampler(next=next, prompt=prompt_tokens, index=1)
txt = tokenizer.detokenize(output_tokens)
print(f'Greedy search generated text: \n{txt}\n')
'\nAs you can see, greedy search starts out making some sense, but quickly starts repeating\nitself. This is a common problem with text generation that can be fixed by some of the\nprobabilistic text generation utilities shown later on!\n'
'\n### Beam search\n\nAt a high-level, beam search keeps track of the `num_beams` most probable sequences at\neach timestep, and predicts the best next token from all sequences. It is an improvement\nover greedy search since it stores more possibilities. However, it is less efficient than\ngreedy search since it has to compute and store multiple potential sequences.\n\n**Note:** beam search with `num_beams=1` is identical to greedy search.\n'
sampler = keras_nlp.samplers.BeamSampler(num_beams=10)
output_tokens = sampler(next=next, prompt=prompt_tokens, index=1)
txt = tokenizer.detokenize(output_tokens)
print(f'Beam search generated text: \n{txt}\n')
'\nSimilar to greedy search, beam search quickly starts repeating itself, since it is still\na deterministic method.\n'
'\n### Random search\n\nRandom search is our first probabilistic method. At each time step, it samples the next\ntoken using the softmax probabilities provided by the model.\n'
sampler = keras_nlp.samplers.RandomSampler()
output_tokens = sampler(next=next, prompt=prompt_tokens, index=1)
txt = tokenizer.detokenize(output_tokens)
print(f'Random search generated text: \n{txt}\n')
'\nVoilà, no repetitions! However, with random search, we may see some nonsensical words\nappearing since any word in the vocabulary has a chance of appearing with this sampling\nmethod. This is fixed by our next search utility, top-k search.\n'
"\n### Top-K search\n\nSimilar to random search, we sample the next token from the probability distribution\nprovided by the model. The only difference is that here, we select out the top `k` most\nprobable tokens, and distribute the probability mass over them before sampling. This way,\nwe won't be sampling from low probability tokens, and hence we would have less\nnonsensical words!\n"
sampler = keras_nlp.samplers.TopKSampler(k=10)
output_tokens = sampler(next=next, prompt=prompt_tokens, index=1)
txt = tokenizer.detokenize(output_tokens)
print(f'Top-K search generated text: \n{txt}\n')
'\n### Top-P search\n\nEven with the top-k search, there is something to improve upon. With top-k search, the\nnumber `k` is fixed, which means it selects the same number of tokens for any probability\ndistribution. Consider two scenarios, one where the probability mass is concentrated over\n2 words and another where the probability mass is evenly concentrated across 10. Should\nwe choose `k=2` or `k=10`? There is no one size that fits all `k` here.\n\nThis is where top-p search comes in! Instead of choosing a `k`, we choose a probability\n`p` that we want the probabilities of the top tokens to sum up to. This way, we can\ndynamically adjust the `k` based on the probability distribution. By setting `p=0.9`, if\n90% of the probability mass is concentrated on the top 2 tokens, we can filter out the\ntop 2 tokens to sample from. If instead the 90% is distributed over 10 tokens, it will\nsimilarly filter out the top 10 tokens to sample from.\n'
sampler = keras_nlp.samplers.TopPSampler(p=0.5)
output_tokens = sampler(next=next, prompt=prompt_tokens, index=1)
txt = tokenizer.detokenize(output_tokens)
print(f'Top-P search generated text: \n{txt}\n')
'\n### Using callbacks for text generation\n\nWe can also wrap the utilities in a callback, which allows you to print out a prediction\nsequence for every epoch of the model! Here is an example of a callback for top-k search:\n'

class TopKTextGenerator(keras.callbacks.Callback):
    """A callback to generate text from a trained model using top-k."""

    def __init__(self, k):
        if False:
            print('Hello World!')
        self.sampler = keras_nlp.samplers.TopKSampler(k)

    def on_epoch_end(self, epoch, logs=None):
        if False:
            i = 10
            return i + 15
        output_tokens = self.sampler(next=next, prompt=prompt_tokens, index=1)
        txt = tokenizer.detokenize(output_tokens)
        print(f'Top-K search generated text: \n{txt}\n')
text_generation_callback = TopKTextGenerator(k=10)
model.fit(train_ds.take(1), verbose=2, epochs=2, callbacks=[text_generation_callback])
'\n## Conclusion\n\nTo recap, in this example, we use KerasNLP layers to train a sub-word vocabulary,\ntokenize training data, create a miniature GPT model, and perform inference with the\ntext generation library.\n\nIf you would like to understand how Transformers work, or learn more about training the\nfull GPT model, here are some further readings:\n\n- Attention Is All You Need [Vaswani et al., 2017](https://arxiv.org/abs/1706.03762)\n- GPT-3 Paper [Brown et al., 2020](https://arxiv.org/abs/2005.14165)\n'