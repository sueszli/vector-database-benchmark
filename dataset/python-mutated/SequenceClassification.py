import os
import cntk as C
import copy

def create_reader(path, is_training, input_dim, label_dim):
    if False:
        return 10
    return C.io.MinibatchSource(C.io.CTFDeserializer(path, C.io.StreamDefs(features=C.io.StreamDef(field='x', shape=input_dim, is_sparse=True), labels=C.io.StreamDef(field='y', shape=label_dim, is_sparse=False))), randomize=is_training, max_sweeps=C.io.INFINITELY_REPEAT if is_training else 1)

def lstm_sequence_classifier(features, num_classes, embedding_dim, LSTM_dim):
    if False:
        while True:
            i = 10
    classifier = C.layers.Sequential([C.layers.Embedding(embedding_dim), C.layers.Recurrence(C.layers.LSTM(LSTM_dim)), C.sequence.last, C.layers.Dense(num_classes)])
    return classifier(features)

def train_sequence_classifier():
    if False:
        for i in range(10):
            print('nop')
    input_dim = 2000
    hidden_dim = 25
    embedding_dim = 50
    num_classes = 5
    features = C.sequence.input_variable(shape=input_dim, is_sparse=True)
    label = C.input_variable(num_classes)
    classifier_output = lstm_sequence_classifier(features, num_classes, embedding_dim, hidden_dim)
    ce = C.cross_entropy_with_softmax(classifier_output, label)
    pe = C.classification_error(classifier_output, label)
    rel_path = '../../../../Tests/EndToEndTests/Text/SequenceClassification/Data/Train.ctf'
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), rel_path)
    reader = create_reader(path, True, input_dim, num_classes)
    input_map = {features: reader.streams.features, label: reader.streams.labels}
    lr_per_sample = C.learning_parameter_schedule_per_sample(0.1)
    progress_printer = C.logging.ProgressPrinter(0)
    trainer = C.Trainer(classifier_output, (ce, pe), C.sgd(classifier_output.parameters, lr=lr_per_sample), progress_printer)
    minibatch_size = 200
    for i in range(251):
        mb = reader.next_minibatch(minibatch_size, input_map=input_map)
        trainer.train_minibatch(mb)
    evaluation_average = copy.copy(trainer.previous_minibatch_evaluation_average)
    loss_average = copy.copy(trainer.previous_minibatch_loss_average)
    return (evaluation_average, loss_average)
if __name__ == '__main__':
    (error, _) = train_sequence_classifier()
    print('Error: %f' % error)