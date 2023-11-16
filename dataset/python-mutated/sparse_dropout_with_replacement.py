from caffe2.python import schema
from caffe2.python.layers.layers import IdList, ModelLayer

class SparseDropoutWithReplacement(ModelLayer):

    def __init__(self, model, input_record, dropout_prob_train, dropout_prob_eval, dropout_prob_predict, replacement_value, name='sparse_dropout', **kwargs):
        if False:
            return 10
        super().__init__(model, name, input_record, **kwargs)
        assert schema.equal_schemas(input_record, IdList), 'Incorrect input type'
        self.dropout_prob_train = float(dropout_prob_train)
        self.dropout_prob_eval = float(dropout_prob_eval)
        self.dropout_prob_predict = float(dropout_prob_predict)
        self.replacement_value = int(replacement_value)
        assert self.dropout_prob_train >= 0 and self.dropout_prob_train <= 1.0, 'Expected 0 <= dropout_prob_train <= 1, but got %s' % self.dropout_prob_train
        assert self.dropout_prob_eval >= 0 and self.dropout_prob_eval <= 1.0, 'Expected 0 <= dropout_prob_eval <= 1, but got %s' % dropout_prob_eval
        assert self.dropout_prob_predict >= 0 and self.dropout_prob_predict <= 1.0, 'Expected 0 <= dropout_prob_predict <= 1, but got %s' % dropout_prob_predict
        assert self.dropout_prob_train > 0 or self.dropout_prob_eval > 0 or self.dropout_prob_predict > 0, 'Ratios all set to 0.0 for train, eval and predict'
        self.output_schema = schema.NewRecord(model.net, IdList)
        if input_record.lengths.metadata:
            self.output_schema.lengths.set_metadata(input_record.lengths.metadata)
        if input_record.items.metadata:
            self.output_schema.items.set_metadata(input_record.items.metadata)

    def _add_ops(self, net, ratio):
        if False:
            while True:
                i = 10
        input_values_blob = self.input_record.items()
        input_lengths_blob = self.input_record.lengths()
        output_lengths_blob = self.output_schema.lengths()
        output_values_blob = self.output_schema.items()
        net.SparseDropoutWithReplacement([input_values_blob, input_lengths_blob], [output_values_blob, output_lengths_blob], ratio=ratio, replacement_value=self.replacement_value)

    def add_train_ops(self, net):
        if False:
            return 10
        self._add_ops(net, self.dropout_prob_train)

    def add_eval_ops(self, net):
        if False:
            i = 10
            return i + 15
        self._add_ops(net, self.dropout_prob_eval)

    def add_ops(self, net):
        if False:
            for i in range(10):
                print('nop')
        self._add_ops(net, self.dropout_prob_predict)