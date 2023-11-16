import sys
import dlib

def sentence_to_vectors(sentence):
    if False:
        print('Hello World!')
    vects = dlib.vectors()
    for word in sentence.split():
        if word[0].isupper():
            vects.append(dlib.vector([1]))
        else:
            vects.append(dlib.vector([0]))
    return vects

def sentence_to_sparse_vectors(sentence):
    if False:
        i = 10
        return i + 15
    vects = dlib.sparse_vectors()
    has_cap = dlib.sparse_vector()
    no_cap = dlib.sparse_vector()
    has_cap.append(dlib.pair(0, 1))
    for word in sentence.split():
        if word[0].isupper():
            vects.append(has_cap)
        else:
            vects.append(no_cap)
    return vects

def print_segment(sentence, names):
    if False:
        i = 10
        return i + 15
    words = sentence.split()
    for name in names:
        for i in name:
            sys.stdout.write(words[i] + ' ')
        sys.stdout.write('\n')
names = dlib.ranges()
segments = dlib.rangess()
sentences = []
sentences.append('The other day I saw a man named Jim Smith')
names.append(dlib.range(8, 10))
segments.append(names)
names.clear()
sentences.append('Davis King is the main author of the dlib Library')
names.append(dlib.range(0, 2))
segments.append(names)
names.clear()
sentences.append('Bob Jones is a name and so is George Clinton')
names.append(dlib.range(0, 2))
names.append(dlib.range(8, 10))
segments.append(names)
names.clear()
sentences.append('My dog is named Bob Barker')
names.append(dlib.range(4, 6))
segments.append(names)
names.clear()
sentences.append('ABC is an acronym but John James Smith is a name')
names.append(dlib.range(5, 8))
segments.append(names)
names.clear()
sentences.append('No names in this sentence at all')
segments.append(names)
names.clear()
use_sparse_vects = False
if use_sparse_vects:
    training_sequences = dlib.sparse_vectorss()
    for s in sentences:
        training_sequences.append(sentence_to_sparse_vectors(s))
else:
    training_sequences = dlib.vectorss()
    for s in sentences:
        training_sequences.append(sentence_to_vectors(s))
params = dlib.segmenter_params()
params.window_size = 3
params.use_high_order_features = True
params.use_BIO_model = True
params.C = 10
model = dlib.train_sequence_segmenter(training_sequences, segments, params)
for (i, s) in enumerate(sentences):
    print_segment(s, model(training_sequences[i]))
test_sentence = 'There once was a man from Nantucket whose name rhymed with Bob Bucket'
if use_sparse_vects:
    print_segment(test_sentence, model(sentence_to_sparse_vectors(test_sentence)))
else:
    print_segment(test_sentence, model(sentence_to_vectors(test_sentence)))
print('Test on training data: {}'.format(dlib.test_sequence_segmenter(model, training_sequences, segments)))
print('Cross validation: {}'.format(dlib.cross_validate_sequence_segmenter(training_sequences, segments, 5, params)))