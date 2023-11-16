"""
Title: Pretraining BERT with Hugging Face Transformers
Author: Sreyan Ghosh
Date created: 2022/07/01
Last modified: 2022/08/27
Description: Pretraining BERT using Hugging Face Transformers on NSP and MLM.
Accelerator: GPU
"""
'\n## Introduction\n'
'\n### BERT (Bidirectional Encoder Representations from Transformers)\n\nIn the field of computer vision, researchers have repeatedly shown the value of\ntransfer learning — pretraining a neural network model on a known task/dataset, for\ninstance ImageNet classification, and then performing fine-tuning — using the trained neural\nnetwork as the basis of a new specific-purpose model. In recent years, researchers\nhave shown that a similar technique can be useful in many natural language tasks.\n\nBERT makes use of Transformer, an attention mechanism that learns contextual relations\nbetween words (or subwords) in a text. In its vanilla form, Transformer includes two\nseparate mechanisms — an encoder that reads the text input and a decoder that produces\na prediction for the task. Since BERT’s goal is to generate a language model, only the\nencoder mechanism is necessary. The detailed workings of Transformer are described in\na paper by Google.\n\nAs opposed to directional models, which read the text input sequentially\n(left-to-right or right-to-left), the Transformer encoder reads the entire\nsequence of words at once. Therefore it is considered bidirectional, though\nit would be more accurate to say that it’s non-directional. This characteristic\nallows the model to learn the context of a word based on all of its surroundings\n(left and right of the word).\n\nWhen training language models, a challenge is defining a prediction goal.\nMany models predict the next word in a sequence (e.g. `"The child came home from _"`),\na directional approach which inherently limits context learning. To overcome this\nchallenge, BERT uses two training strategies:\n\n### Masked Language Modeling (MLM)\n\nBefore feeding word sequences into BERT, 15% of the words in each sequence are replaced\nwith a `[MASK]` token. The model then attempts to predict the original value of the masked\nwords, based on the context provided by the other, non-masked, words in the sequence.\n\n### Next Sentence Prediction (NSP)\n\nIn the BERT training process, the model receives pairs of sentences as input and learns to\npredict if the second sentence in the pair is the subsequent sentence in the original\ndocument. During training, 50% of the inputs are a pair in which the second sentence is the\nsubsequent sentence in the original document, while in the other 50% a random sentence\nfrom the corpus is chosen as the second sentence. The assumption is that the random sentence\nwill represent a disconnect from the first sentence.\n\nThough Google provides a pretrained BERT checkpoint for English, you may often need\nto either pretrain the model from scratch for a different language, or do a\ncontinued-pretraining to fit the model to a new domain. In this notebook, we pretrain\nBERT from scratch optimizing both MLM and NSP objectves using 🤗 Transformers on the `WikiText`\nEnglish dataset loaded from 🤗 Datasets.\n'
'\n## Setup\n'
'\n### Installing the requirements\n'
'shell\npip install git+https://github.com/huggingface/transformers.git\npip install datasets\npip install huggingface-hub\npip install nltk\n'
'\n### Importing the necessary libraries\n'
import nltk
import random
import logging
import keras
nltk.download('punkt')
keras.utils.set_random_seed(42)
'\n### Define certain variables\n'
TOKENIZER_BATCH_SIZE = 256
TOKENIZER_VOCABULARY = 25000
BLOCK_SIZE = 128
NSP_PROB = 0.5
SHORT_SEQ_PROB = 0.1
MAX_LENGTH = 512
MLM_PROB = 0.2
TRAIN_BATCH_SIZE = 2
MAX_EPOCHS = 1
LEARNING_RATE = 0.0001
MODEL_CHECKPOINT = 'bert-base-cased'
'\n## Load the WikiText dataset\n'
'\nWe now download the `WikiText` language modeling dataset. It is a collection of over\n100 million tokens extracted from the set of verified "Good" and "Featured" articles on\nWikipedia.\n\nWe load the dataset from [🤗 Datasets](https://github.com/huggingface/datasets).\nFor the purpose of demonstration in this notebook, we work with only the `train`\nsplit of the dataset. This can be easily done with the `load_dataset` function.\n'
from datasets import load_dataset
dataset = load_dataset('wikitext', 'wikitext-2-raw-v1')
'\nThe dataset just has one column which is the raw text, and this is all we need for\npretraining BERT!\n'
print(dataset)
'\n## Training a new Tokenizer\n'
"\nFirst we train our own tokenizer from scratch on our corpus, so that can we\ncan use it to train our language model from scratch.\n\nBut why would you need to train a tokenizer? That's because Transformer models very\noften use subword tokenization algorithms, and they need to be trained to identify the\nparts of words that are often present in the corpus you are using.\n\nThe 🤗 Transformers `Tokenizer` (as the name indicates) will tokenize the inputs\n(including converting the tokens to their corresponding IDs in the pretrained vocabulary)\nand put it in a format the model expects, as well as generate the other inputs that model\nrequires.\n\nFirst we make a list of all the raw documents from the `WikiText` corpus:\n"
all_texts = [doc for doc in dataset['train']['text'] if len(doc) > 0 and (not doc.startswith(' ='))]
'\nNext we make a `batch_iterator` function that will aid us to train our tokenizer.\n'

def batch_iterator():
    if False:
        i = 10
        return i + 15
    for i in range(0, len(all_texts), TOKENIZER_BATCH_SIZE):
        yield all_texts[i:i + TOKENIZER_BATCH_SIZE]
'\nIn this notebook, we train a tokenizer with the exact same algorithms and\nparameters as an existing one. For instance, we train a new version of the\n`BERT-CASED` tokenzier on `Wikitext-2` using the same tokenization algorithm.\n\nFirst we need to load the tokenizer we want to use as a model:\n'
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)
'\nNow we train our tokenizer using the entire `train` split of the `Wikitext-2`\ndataset.\n'
tokenizer = tokenizer.train_new_from_iterator(batch_iterator(), vocab_size=TOKENIZER_VOCABULARY)
'\nSo now we our done training our new tokenizer! Next we move on to the data\npre-processing steps.\n'
'\n## Data Pre-processing\n'
'\nFor the sake of demonstrating the workflow, in this notebook we only take\nsmall subsets of the entire WikiText `train` and `test` splits.\n'
dataset['train'] = dataset['train'].select([i for i in range(1000)])
dataset['validation'] = dataset['validation'].select([i for i in range(1000)])
"\nBefore we can feed those texts to our model, we need to pre-process them and get them\nready for the task. As mentioned earlier, the BERT pretraining task includes two tasks\nin total, the `NSP` task and the `MLM` task. 🤗 Transformers have an easy to implement\n`collator` called the `DataCollatorForLanguageModeling`. However, we need to get the\ndata ready for `NSP` manually.\n\nNext we write a simple function called the `prepare_train_features` that helps us in\nthe pre-processing and is compatible with 🤗 Datasets. To summarize, our pre-processing\nfunction should:\n\n- Get the dataset ready for the NSP task by creating pairs of sentences (A,B), where B\neither actually follows A, or B is randomly sampled from somewhere else in the corpus.\nIt should also generate a corresponding label for each pair, which is 1 if B actually\nfollows A and 0 if not.\n- Tokenize the text dataset into it's corresponding token ids that will be used for\nembedding look-up in BERT\n- Create additional inputs for the model like `token_type_ids`, `attention_mask`, etc.\n"
max_num_tokens = BLOCK_SIZE - tokenizer.num_special_tokens_to_add(pair=True)

def prepare_train_features(examples):
    if False:
        while True:
            i = 10
    'Function to prepare features for NSP task\n\n    Arguments:\n      examples: A dictionary with 1 key ("text")\n        text: List of raw documents (str)\n    Returns:\n      examples:  A dictionary with 4 keys\n        input_ids: List of tokenized, concatnated, and batched\n          sentences from the individual raw documents (int)\n        token_type_ids: List of integers (0 or 1) corresponding\n          to: 0 for senetence no. 1 and padding, 1 for sentence\n          no. 2\n        attention_mask: List of integers (0 or 1) corresponding\n          to: 1 for non-padded tokens, 0 for padded\n        next_sentence_label: List of integers (0 or 1) corresponding\n          to: 1 if the second sentence actually follows the first,\n          0 if the senetence is sampled from somewhere else in the corpus\n    '
    examples['document'] = [d.strip() for d in examples['text'] if len(d) > 0 and (not d.startswith(' ='))]
    examples['sentences'] = [nltk.tokenize.sent_tokenize(document) for document in examples['document']]
    examples['tokenized_sentences'] = [[tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sent)) for sent in doc] for doc in examples['sentences']]
    examples['input_ids'] = []
    examples['token_type_ids'] = []
    examples['attention_mask'] = []
    examples['next_sentence_label'] = []
    for (doc_index, document) in enumerate(examples['tokenized_sentences']):
        current_chunk = []
        current_length = 0
        i = 0
        target_seq_length = max_num_tokens
        if random.random() < SHORT_SEQ_PROB:
            target_seq_length = random.randint(2, max_num_tokens)
        while i < len(document):
            segment = document[i]
            current_chunk.append(segment)
            current_length += len(segment)
            if i == len(document) - 1 or current_length >= target_seq_length:
                if current_chunk:
                    a_end = 1
                    if len(current_chunk) >= 2:
                        a_end = random.randint(1, len(current_chunk) - 1)
                    tokens_a = []
                    for j in range(a_end):
                        tokens_a.extend(current_chunk[j])
                    tokens_b = []
                    if len(current_chunk) == 1 or random.random() < NSP_PROB:
                        is_random_next = True
                        target_b_length = target_seq_length - len(tokens_a)
                        for _ in range(10):
                            random_document_index = random.randint(0, len(examples['tokenized_sentences']) - 1)
                            if random_document_index != doc_index:
                                break
                        random_document = examples['tokenized_sentences'][random_document_index]
                        random_start = random.randint(0, len(random_document) - 1)
                        for j in range(random_start, len(random_document)):
                            tokens_b.extend(random_document[j])
                            if len(tokens_b) >= target_b_length:
                                break
                        num_unused_segments = len(current_chunk) - a_end
                        i -= num_unused_segments
                    else:
                        is_random_next = False
                        for j in range(a_end, len(current_chunk)):
                            tokens_b.extend(current_chunk[j])
                    input_ids = tokenizer.build_inputs_with_special_tokens(tokens_a, tokens_b)
                    token_type_ids = tokenizer.create_token_type_ids_from_sequences(tokens_a, tokens_b)
                    padded = tokenizer.pad({'input_ids': input_ids, 'token_type_ids': token_type_ids}, padding='max_length', max_length=MAX_LENGTH)
                    examples['input_ids'].append(padded['input_ids'])
                    examples['token_type_ids'].append(padded['token_type_ids'])
                    examples['attention_mask'].append(padded['attention_mask'])
                    examples['next_sentence_label'].append(1 if is_random_next else 0)
                    current_chunk = []
                    current_length = 0
            i += 1
    del examples['document']
    del examples['sentences']
    del examples['text']
    del examples['tokenized_sentences']
    return examples
tokenized_dataset = dataset.map(prepare_train_features, batched=True, remove_columns=['text'], num_proc=1)
"\nFor MLM we are going to use the same preprocessing as before for our dataset with\none additional step: we randomly mask some tokens (by replacing them by [MASK])\nand the labels will be adjusted to only include the masked tokens\n(we don't have to predict the non-masked tokens). If you use a tokenizer you trained\nyourself, make sure the [MASK] token is among the special tokens you passed during training!\n\nTo get the data ready for MLM, we simply use the `collator` called the\n`DataCollatorForLanguageModeling` provided by the 🤗 Transformers library on our dataset\nthat is already ready for the NSP task. The `collator` expects certain parameters.\nWe use the default ones from the original BERT paper in this notebook. The\n`return_tensors='tf'` ensures that we get `tf.Tensor` objects back.\n"
from transformers import DataCollatorForLanguageModeling
collater = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=MLM_PROB, return_tensors='tf')
'\nNext we define our training set with which we train our model. Again, 🤗 Datasets\nprovides us with the `to_tf_dataset` method which will help us integrate our dataset with\nthe `collator` defined above. The method expects certain parameters:\n\n- **columns**: the columns which will serve as our independant variables\n- **label_cols**: the columns which will serve as our labels or dependant variables\n- **batch_size**: our batch size for training\n- **shuffle**: whether we want to shuffle our training dataset\n- **collate_fn**: our collator function\n'
train = tokenized_dataset['train'].to_tf_dataset(columns=['input_ids', 'token_type_ids', 'attention_mask'], label_cols=['labels', 'next_sentence_label'], batch_size=TRAIN_BATCH_SIZE, shuffle=True, collate_fn=collater)
validation = tokenized_dataset['validation'].to_tf_dataset(columns=['input_ids', 'token_type_ids', 'attention_mask'], label_cols=['labels', 'next_sentence_label'], batch_size=TRAIN_BATCH_SIZE, shuffle=True, collate_fn=collater)
'\n## Defining the model\n'
'\nTo define our model, first we need to define a config which will help us define certain\nparameters of our model architecture. This includes parameters like number of transformer\nlayers, number of attention heads, hidden dimension, etc. For this notebook, we try\nto define the exact config defined in the original BERT paper.\n\nWe can easily achieve this using the `BertConfig` class from the 🤗 Transformers library.\nThe `from_pretrained()` method expects the name of a model. Here we define the simplest\nmodel with which we also trained our model, i.e., `bert-base-cased`.\n'
from transformers import BertConfig
config = BertConfig.from_pretrained(MODEL_CHECKPOINT)
'\nFor defining our model we use the `TFBertForPreTraining` class from the 🤗 Transformers\nlibrary. This class internally handles everything starting from defining our model, to\nunpacking our inputs and calculating the loss. So we need not do anything ourselves except\ndefining the model with the correct `config` we want!\n'
from transformers import TFBertForPreTraining
model = TFBertForPreTraining(config)
'\nNow we define our optimizer and compile the model. The loss calculation is handled\ninternally and so we need not worry about that!\n'
from keras.optimizers import Adam
model.compile(optimizer=Adam(learning_rate=LEARNING_RATE))
'\nFinally all steps are done and now we can start training our model!\n'
model.fit(train, validation_data=validation, epochs=MAX_EPOCHS)
'\nOur model has now been trained! We suggest to please train the model on the complete\ndataset for atleast 50 epochs for decent performance. The pretrained model now acts as\na language model and is meant to be fine-tuned on a downstream task. Thus it can now be\nfine-tuned on any downstream task like Question Answering, Text Classification\netc.!\n'
'\nNow you can push this model to 🤗 Model Hub and also share it with with all your friends,\nfamily, favorite pets: they can all load it with the identifier\n`"your-username/the-name-you-picked"` so for instance:\n\n```python\nmodel.push_to_hub("pretrained-bert", organization="keras-io")\ntokenizer.push_to_hub("pretrained-bert", organization="keras-io")\n```\nAnd after you push your model this is how you can load it in the future!\n\n```python\nfrom transformers import TFBertForPreTraining\n\nmodel = TFBertForPreTraining.from_pretrained("your-username/my-awesome-model")\n```\nor, since it\'s a pretrained model and you would generally use it for fine-tuning\non a downstream task, you can also load it for some other task like:\n\n```python\nfrom transformers import TFBertForSequenceClassification\n\nmodel = TFBertForSequenceClassification.from_pretrained("your-username/my-awesome-model")\n```\nIn this case, the pretraining head will be dropped and the model will just be initialized\nwith the transformer layers. A new task-specific head will be added with random weights.\n'