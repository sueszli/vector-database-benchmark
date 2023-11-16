"""
Title: English-to-Spanish translation with KerasNLP
Author: [Abheesht Sharma](https://github.com/abheesht17/)
Date created: 2022/05/26
Last modified: 2022/12/21
Description: Use KerasNLP to train a sequence-to-sequence Transformer model on the machine translation task.
Accelerator: GPU
"""
"\n## Introduction\n\nKerasNLP provides building blocks for NLP (model layers, tokenizers, metrics, etc.) and\nmakes it convenient to construct NLP pipelines.\n\nIn this example, we'll use KerasNLP layers to build an encoder-decoder Transformer\nmodel, and train it on the English-to-Spanish machine translation task.\n\nThis example is based on the\n[English-to-Spanish NMT\nexample](https://keras.io/examples/nlp/neural_machine_translation_with_transformer/)\nby [fchollet](https://twitter.com/fchollet). The original example is more low-level\nand implements layers from scratch, whereas this example uses KerasNLP to show\nsome more advanced approaches, such as subword tokenization and using metrics\nto compute the quality of generated translations.\n\nYou'll learn how to:\n\n- Tokenize text using `keras_nlp.tokenizers.WordPieceTokenizer`.\n- Implement a sequence-to-sequence Transformer model using KerasNLP's\n`keras_nlp.layers.TransformerEncoder`, `keras_nlp.layers.TransformerDecoder` and\n`keras_nlp.layers.TokenAndPositionEmbedding` layers, and train it.\n- Use `keras_nlp.samplers` to generate translations of unseen input sentences\n using the top-p decoding strategy!\n\nDon't worry if you aren't familiar with KerasNLP. This tutorial will start with\nthe basics. Let's dive right in!\n"
"\n## Setup\n\nBefore we start implementing the pipeline, let's import all the libraries we need.\n"
'shell\n!pip install -q rouge-score\n!pip install -q git+https://github.com/keras-team/keras-nlp.git --upgrade\n'
import keras_nlp
import pathlib
import random
import keras
from keras import ops
import tensorflow.data as tf_data
from tensorflow_text.tools.wordpiece_vocab import bert_vocab_from_dataset as bert_vocab
"\nLet's also define our parameters/hyperparameters.\n"
BATCH_SIZE = 64
EPOCHS = 1
MAX_SEQUENCE_LENGTH = 40
ENG_VOCAB_SIZE = 15000
SPA_VOCAB_SIZE = 15000
EMBED_DIM = 256
INTERMEDIATE_DIM = 2048
NUM_HEADS = 8
"\n## Downloading the data\n\nWe'll be working with an English-to-Spanish translation dataset\nprovided by [Anki](https://www.manythings.org/anki/). Let's download it:\n"
text_file = keras.utils.get_file(fname='spa-eng.zip', origin='http://storage.googleapis.com/download.tensorflow.org/data/spa-eng.zip', extract=True)
text_file = pathlib.Path(text_file).parent / 'spa-eng' / 'spa.txt'
'\n## Parsing the data\n\nEach line contains an English sentence and its corresponding Spanish sentence.\nThe English sentence is the *source sequence* and Spanish one is the *target sequence*.\nBefore adding the text to a list, we convert it to lowercase.\n'
with open(text_file) as f:
    lines = f.read().split('\n')[:-1]
text_pairs = []
for line in lines:
    (eng, spa) = line.split('\t')
    eng = eng.lower()
    spa = spa.lower()
    text_pairs.append((eng, spa))
"\nHere's what our sentence pairs look like:\n"
for _ in range(5):
    print(random.choice(text_pairs))
"\nNow, let's split the sentence pairs into a training set, a validation set,\nand a test set.\n"
random.shuffle(text_pairs)
num_val_samples = int(0.15 * len(text_pairs))
num_train_samples = len(text_pairs) - 2 * num_val_samples
train_pairs = text_pairs[:num_train_samples]
val_pairs = text_pairs[num_train_samples:num_train_samples + num_val_samples]
test_pairs = text_pairs[num_train_samples + num_val_samples:]
print(f'{len(text_pairs)} total pairs')
print(f'{len(train_pairs)} training pairs')
print(f'{len(val_pairs)} validation pairs')
print(f'{len(test_pairs)} test pairs')
"\n## Tokenizing the data\n\nWe'll define two tokenizers - one for the source language (English), and the other\nfor the target language (Spanish). We'll be using\n`keras_nlp.tokenizers.WordPieceTokenizer` to tokenize the text.\n`keras_nlp.tokenizers.WordPieceTokenizer` takes a WordPiece vocabulary\nand has functions for tokenizing the text, and detokenizing sequences of tokens.\n\nBefore we define the two tokenizers, we first need to train them on the dataset\nwe have. The WordPiece tokenization algorithm is a subword tokenization algorithm;\ntraining it on a corpus gives us a vocabulary of subwords. A subword tokenizer\nis a compromise between word tokenizers (word tokenizers need very large\nvocabularies for good coverage of input words), and character tokenizers\n(characters don't really encode meaning like words do). Luckily, KerasNLP\nmakes it very simple to train WordPiece on a corpus with the\n`keras_nlp.tokenizers.compute_word_piece_vocabulary` utility.\n"

def train_word_piece(text_samples, vocab_size, reserved_tokens):
    if False:
        return 10
    word_piece_ds = tf_data.Dataset.from_tensor_slices(text_samples)
    vocab = keras_nlp.tokenizers.compute_word_piece_vocabulary(word_piece_ds.batch(1000).prefetch(2), vocabulary_size=vocab_size, reserved_tokens=reserved_tokens)
    return vocab
'\nEvery vocabulary has a few special, reserved tokens. We have four such tokens:\n\n- `"[PAD]"` - Padding token. Padding tokens are appended to the input sequence\nlength when the input sequence length is shorter than the maximum sequence length.\n- `"[UNK]"` - Unknown token.\n- `"[START]"` - Token that marks the start of the input sequence.\n- `"[END]"` - Token that marks the end of the input sequence.\n'
reserved_tokens = ['[PAD]', '[UNK]', '[START]', '[END]']
eng_samples = [text_pair[0] for text_pair in train_pairs]
eng_vocab = train_word_piece(eng_samples, ENG_VOCAB_SIZE, reserved_tokens)
spa_samples = [text_pair[1] for text_pair in train_pairs]
spa_vocab = train_word_piece(spa_samples, SPA_VOCAB_SIZE, reserved_tokens)
"\nLet's see some tokens!\n"
print('English Tokens: ', eng_vocab[100:110])
print('Spanish Tokens: ', spa_vocab[100:110])
"\nNow, let's define the tokenizers. We will configure the tokenizers with the\nthe vocabularies trained above.\n"
eng_tokenizer = keras_nlp.tokenizers.WordPieceTokenizer(vocabulary=eng_vocab, lowercase=False)
spa_tokenizer = keras_nlp.tokenizers.WordPieceTokenizer(vocabulary=spa_vocab, lowercase=False)
"\nLet's try and tokenize a sample from our dataset! To verify whether the text has\nbeen tokenized correctly, we can also detokenize the list of tokens back to the\noriginal text.\n"
eng_input_ex = text_pairs[0][0]
eng_tokens_ex = eng_tokenizer.tokenize(eng_input_ex)
print('English sentence: ', eng_input_ex)
print('Tokens: ', eng_tokens_ex)
print('Recovered text after detokenizing: ', eng_tokenizer.detokenize(eng_tokens_ex))
print()
spa_input_ex = text_pairs[0][1]
spa_tokens_ex = spa_tokenizer.tokenize(spa_input_ex)
print('Spanish sentence: ', spa_input_ex)
print('Tokens: ', spa_tokens_ex)
print('Recovered text after detokenizing: ', spa_tokenizer.detokenize(spa_tokens_ex))
'\n## Format datasets\n\nNext, we\'ll format our datasets.\n\nAt each training step, the model will seek to predict target words N+1 (and beyond)\nusing the source sentence and the target words 0 to N.\n\nAs such, the training dataset will yield a tuple `(inputs, targets)`, where:\n\n- `inputs` is a dictionary with the keys `encoder_inputs` and `decoder_inputs`.\n`encoder_inputs` is the tokenized source sentence and `decoder_inputs` is the target\nsentence "so far",\nthat is to say, the words 0 to N used to predict word N+1 (and beyond) in the target\nsentence.\n- `target` is the target sentence offset by one step:\nit provides the next words in the target sentence -- what the model will try to predict.\n\nWe will add special tokens, `"[START]"` and `"[END]"`, to the input Spanish\nsentence after tokenizing the text. We will also pad the input to a fixed length.\nThis can be easily done using `keras_nlp.layers.StartEndPacker`.\n'

def preprocess_batch(eng, spa):
    if False:
        i = 10
        return i + 15
    batch_size = ops.shape(spa)[0]
    eng = eng_tokenizer(eng)
    spa = spa_tokenizer(spa)
    eng_start_end_packer = keras_nlp.layers.StartEndPacker(sequence_length=MAX_SEQUENCE_LENGTH, pad_value=eng_tokenizer.token_to_id('[PAD]'))
    eng = eng_start_end_packer(eng)
    spa_start_end_packer = keras_nlp.layers.StartEndPacker(sequence_length=MAX_SEQUENCE_LENGTH + 1, start_value=spa_tokenizer.token_to_id('[START]'), end_value=spa_tokenizer.token_to_id('[END]'), pad_value=spa_tokenizer.token_to_id('[PAD]'))
    spa = spa_start_end_packer(spa)
    return ({'encoder_inputs': eng, 'decoder_inputs': spa[:, :-1]}, spa[:, 1:])

def make_dataset(pairs):
    if False:
        while True:
            i = 10
    (eng_texts, spa_texts) = zip(*pairs)
    eng_texts = list(eng_texts)
    spa_texts = list(spa_texts)
    dataset = tf_data.Dataset.from_tensor_slices((eng_texts, spa_texts))
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.map(preprocess_batch, num_parallel_calls=tf_data.AUTOTUNE)
    return dataset.shuffle(2048).prefetch(16).cache()
train_ds = make_dataset(train_pairs)
val_ds = make_dataset(val_pairs)
"\nLet's take a quick look at the sequence shapes\n(we have batches of 64 pairs, and all sequences are 40 steps long):\n"
for (inputs, targets) in train_ds.take(1):
    print(f"""inputs["encoder_inputs"].shape: {inputs['encoder_inputs'].shape}""")
    print(f"""inputs["decoder_inputs"].shape: {inputs['decoder_inputs'].shape}""")
    print(f'targets.shape: {targets.shape}')
'\n## Building the model\n\nNow, let\'s move on to the exciting part - defining our model!\nWe first need an embedding layer, i.e., a vector for every token in our input sequence.\nThis embedding layer can be initialised randomly. We also need a positional\nembedding layer which encodes the word order in the sequence. The convention is\nto add these two embeddings. KerasNLP has a `keras_nlp.layers.TokenAndPositionEmbedding `\nlayer which does all of the above steps for us.\n\nOur sequence-to-sequence Transformer consists of a `keras_nlp.layers.TransformerEncoder`\nlayer and a `keras_nlp.layers.TransformerDecoder` layer chained together.\n\nThe source sequence will be passed to `keras_nlp.layers.TransformerEncoder`, which\nwill produce a new representation of it. This new representation will then be passed\nto the `keras_nlp.layers.TransformerDecoder`, together with the target sequence\nso far (target words 0 to N). The `keras_nlp.layers.TransformerDecoder` will\nthen seek to predict the next words in the target sequence (N+1 and beyond).\n\nA key detail that makes this possible is causal masking.\nThe `keras_nlp.layers.TransformerDecoder` sees the entire sequence at once, and\nthus we must make sure that it only uses information from target tokens 0 to N\nwhen predicting token N+1 (otherwise, it could use information from the future,\nwhich would result in a model that cannot be used at inference time). Causal masking\nis enabled by default in `keras_nlp.layers.TransformerDecoder`.\n\nWe also need to mask the padding tokens (`"[PAD]"`). For this, we can set the\n`mask_zero` argument of the `keras_nlp.layers.TokenAndPositionEmbedding` layer\nto True. This will then be propagated to all subsequent layers.\n'
encoder_inputs = keras.Input(shape=(None,), dtype='int64', name='encoder_inputs')
x = keras_nlp.layers.TokenAndPositionEmbedding(vocabulary_size=ENG_VOCAB_SIZE, sequence_length=MAX_SEQUENCE_LENGTH, embedding_dim=EMBED_DIM)(encoder_inputs)
encoder_outputs = keras_nlp.layers.TransformerEncoder(intermediate_dim=INTERMEDIATE_DIM, num_heads=NUM_HEADS)(inputs=x)
encoder = keras.Model(encoder_inputs, encoder_outputs)
decoder_inputs = keras.Input(shape=(None,), dtype='int64', name='decoder_inputs')
encoded_seq_inputs = keras.Input(shape=(None, EMBED_DIM), name='decoder_state_inputs')
x = keras_nlp.layers.TokenAndPositionEmbedding(vocabulary_size=SPA_VOCAB_SIZE, sequence_length=MAX_SEQUENCE_LENGTH, embedding_dim=EMBED_DIM)(decoder_inputs)
x = keras_nlp.layers.TransformerDecoder(intermediate_dim=INTERMEDIATE_DIM, num_heads=NUM_HEADS)(decoder_sequence=x, encoder_sequence=encoded_seq_inputs)
x = keras.layers.Dropout(0.5)(x)
decoder_outputs = keras.layers.Dense(SPA_VOCAB_SIZE, activation='softmax')(x)
decoder = keras.Model([decoder_inputs, encoded_seq_inputs], decoder_outputs)
decoder_outputs = decoder([decoder_inputs, encoder_outputs])
transformer = keras.Model([encoder_inputs, decoder_inputs], decoder_outputs, name='transformer')
"\n## Training our model\n\nWe'll use accuracy as a quick way to monitor training progress on the validation data.\nNote that machine translation typically uses BLEU scores as well as other metrics,\nrather than accuracy. However, in order to use metrics like ROUGE, BLEU, etc. we\nwill have decode the probabilities and generate the text. Text generation is\ncomputationally expensive, and performing this during training is not recommended.\n\nHere we only train for 1 epoch, but to get the model to actually converge\nyou should train for at least 10 epochs.\n"
transformer.summary()
transformer.compile('rmsprop', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
transformer.fit(train_ds, epochs=EPOCHS, validation_data=val_ds)
'\n## Decoding test sentences (qualitative analysis)\n\nFinally, let\'s demonstrate how to translate brand new English sentences.\nWe simply feed into the model the tokenized English sentence\nas well as the target token `"[START]"`. The model outputs probabilities of the\nnext token. We then we repeatedly generated the next token conditioned on the\ntokens generated so far, until we hit the token `"[END]"`.\n\nFor decoding, we will use the `keras_nlp.samplers` module from\nKerasNLP. Greedy Decoding is a text decoding method which outputs the most\nlikely next token at each time step, i.e., the token with the highest probability.\n'

def decode_sequences(input_sentences):
    if False:
        for i in range(10):
            print('nop')
    batch_size = 1
    encoder_input_tokens = ops.convert_to_tensor(eng_tokenizer(input_sentences))
    if len(encoder_input_tokens[0]) < MAX_SEQUENCE_LENGTH:
        pads = ops.full((1, MAX_SEQUENCE_LENGTH - len(encoder_input_tokens[0])), 0)
        encoder_input_tokens = ops.concatenate([encoder_input_tokens, pads], 1)

    def next(prompt, cache, index):
        if False:
            print('Hello World!')
        logits = transformer([encoder_input_tokens, prompt])[:, index - 1, :]
        hidden_states = None
        return (logits, hidden_states, cache)
    length = 40
    start = ops.full((batch_size, 1), spa_tokenizer.token_to_id('[START]'))
    pad = ops.full((batch_size, length - 1), spa_tokenizer.token_to_id('[PAD]'))
    prompt = ops.concatenate((start, pad), axis=-1)
    generated_tokens = keras_nlp.samplers.GreedySampler()(next, prompt, end_token_id=spa_tokenizer.token_to_id('[END]'), index=1)
    generated_sentences = spa_tokenizer.detokenize(generated_tokens)
    return generated_sentences
test_eng_texts = [pair[0] for pair in test_pairs]
for i in range(2):
    input_sentence = random.choice(test_eng_texts)
    translated = decode_sequences([input_sentence])
    translated = translated.numpy()[0].decode('utf-8')
    translated = translated.replace('[PAD]', '').replace('[START]', '').replace('[END]', '').strip()
    print(f'** Example {i} **')
    print(input_sentence)
    print(translated)
    print()
"\n## Evaluating our model (quantitative analysis)\n\nThere are many metrics which are used for text generation tasks. Here, to\nevaluate translations generated by our model, let's compute the ROUGE-1 and\nROUGE-2 scores. Essentially, ROUGE-N is a score based on the number of common\nn-grams between the reference text and the generated text. ROUGE-1 and ROUGE-2\nuse the number of common unigrams and bigrams, respectively.\n\nWe will calculate the score over 30 test samples (since decoding is an\nexpensive process).\n"
rouge_1 = keras_nlp.metrics.RougeN(order=1)
rouge_2 = keras_nlp.metrics.RougeN(order=2)
for test_pair in test_pairs[:30]:
    input_sentence = test_pair[0]
    reference_sentence = test_pair[1]
    translated_sentence = decode_sequences([input_sentence])
    translated_sentence = translated_sentence.numpy()[0].decode('utf-8')
    translated_sentence = translated_sentence.replace('[PAD]', '').replace('[START]', '').replace('[END]', '').strip()
    rouge_1(reference_sentence, translated_sentence)
    rouge_2(reference_sentence, translated_sentence)
print('ROUGE-1 Score: ', rouge_1.result())
print('ROUGE-2 Score: ', rouge_2.result())
'\nAfter 10 epochs, the scores are as follows:\n\n|               | **ROUGE-1** | **ROUGE-2** |\n|:-------------:|:-----------:|:-----------:|\n| **Precision** |    0.568    |    0.374    |\n|   **Recall**  |    0.615    |    0.394    |\n|  **F1 Score** |    0.579    |    0.381    |\n'