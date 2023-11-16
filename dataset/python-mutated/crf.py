import numpy as np
from caffe2.python import brew, core, model_helper, recurrent
'\nDue to a limitation in ReccurentNetworkOp, this layer only supports batch_size=1\nIn order to support batch_size > 1, we will have to implement the CRFUnit\nand its gradient in C++ and handle the different batches there.\n'

class CRFWithLoss:

    def __init__(self, model, num_classes, transitions_blob=None):
        if False:
            i = 10
            return i + 15
        self.model = model
        self.num_classes = num_classes
        self.num_classes_padded = num_classes + 2
        if not transitions_blob:
            transitions_blob = self.model.param_init_net.UniformFill([], [core.ScopedBlobReference('crf_transitions')], shape=[self.num_classes_padded, self.num_classes_padded], min=-1.0, max=1.0)
        self.transitions = transitions_blob
        self.model.params.append(self.transitions)

    def crf_loss(self, predictions, labels, seq_lengths=None):
        if False:
            while True:
                i = 10
        transitions_snapshot = self.model.net.Copy(self.transitions, core.ScopedBlobReference('transitions_snapshot'))
        path_unary_score = self._gather_entries_sum(predictions, labels, self.num_classes)
        predictions = CRFWithLoss.pad_predictions(predictions, self.model.param_init_net, self.model.net, self.num_classes)
        labels = CRFWithLoss.pad_labels(labels, self.model.param_init_net, self.model.net, self.num_classes)
        path_binary_score = self._path_binary_scores(labels, transitions_snapshot, seq_lengths)
        path_total_score = self.model.net.Add([path_binary_score, path_unary_score], core.ScopedBlobReference('path_total'))
        zero_index = self.model.param_init_net.ConstantFill([], shape=[1], value=0)
        initial_state = self.model.net.Gather([predictions, zero_index], core.ScopedBlobReference('rnn_initial'), dense_gradient=True)
        (input_data, _) = self.model.net.RemovePadding([predictions], padding_width=1, end_padding_width=0, outputs=2)
        input_data = self.model.net.ExpandDims([input_data], core.ScopedBlobReference('rnn_input_data'), dims=[1])
        transitions_copy = self.model.net.Copy(transitions_snapshot, core.ScopedBlobReference('transitions_copy'))
        all_paths_scores = self._crf_forward(input_data, initial_state, transitions_copy)
        loss = self.model.net.Sub([all_paths_scores, path_total_score], core.ScopedBlobReference('crf_loss'))
        return loss

    def _path_binary_scores(self, labels, transitions, seq_lengths=None):
        if False:
            while True:
                i = 10
        (column_ids, _) = self.model.net.RemovePadding([labels], outputs=2, padding_width=1, end_padding_width=0)
        (row_ids, _) = self.model.net.RemovePadding([labels], outputs=2, padding_width=0, end_padding_width=1)
        num_columns_blob = self.model.net.ConstantFill([row_ids], value=self.num_classes_padded)
        flattened_ids = self.model.net.Mul([row_ids, num_columns_blob])
        flattened_ids = self.model.net.Add([flattened_ids, column_ids])
        flattened_transitions = self.model.net.FlattenToVec([transitions])
        entries = self.model.net.Gather([flattened_transitions, flattened_ids], dense_gradient=True)
        return self.model.ReduceFrontSum(entries)

    def _gather_entries_sum(self, in_data, indices, index_size):
        if False:
            print('Hello World!')
        indices = self.model.net.Cast([indices], to='int64')
        index_size_blob = self.model.param_init_net.ConstantFill([], shape=[1], value=index_size)
        query_one_hot = self.model.net.OneHot([indices, index_size_blob])
        flattend_query = self.model.net.FlattenToVec(query_one_hot)
        flattend_data = self.model.net.FlattenToVec(in_data)
        query_scores = self.model.net.DotProduct([flattend_query, flattend_data])
        final_sum = self.model.net.ReduceFrontSum([query_scores])
        return final_sum

    def _crf_forward(self, input_blob, initial_state, transitions_copy, seq_lengths=None):
        if False:
            print('Hello World!')
        out_last = self.build_crf_net(input_blob, initial_state, transitions_copy)
        (out_last, _) = self.model.net.Reshape([out_last], outputs=2, shape=(self.num_classes_padded,))
        zero_segment_id = self.model.param_init_net.ConstantFill([], value=0, shape=[self.num_classes_padded], dtype=core.DataType.INT32)
        accum_score = self.model.net.SortedSegmentRangeLogSumExp([out_last, zero_segment_id])
        (accum_score, _) = self.model.net.Reshape(accum_score, outputs=2, shape=())
        return accum_score

    def build_crf_net(self, input_blob, initial_state, transitions):
        if False:
            while True:
                i = 10
        '\n            Adds the crf_net recurrent operator to the model.\n\n            model: model_helper.ModelHelper object new operators would be added\n            to\n\n            input_blob: the input sequence in a format T x N x D\n            where T is sequence size, N - batch size and D - input dimension\n            ##Only supports batch-size 1##\n\n            seq_lengths: blob containing sequence lengths (unused)\n            '
        scope = 'crf_net'

        def s(name):
            if False:
                i = 10
                return i + 15
            ''
            return '{}/{}'.format(str(scope), str(name))
        step_model = model_helper.ModelHelper(name='crf_step', param_model=self.model)
        (input_t, cell_t_prev, _) = step_model.net.AddExternalInputs(core.ScopedBlobReference('input_t'), core.ScopedBlobReference('cell_t_prev'), transitions)
        zero_segment_id = step_model.param_init_net.ConstantFill([], [s('zero_segment_id')], value=0, shape=[self.num_classes_padded], dtype=core.DataType.INT32)
        step_model.param_init_net.AddExternalOutput(zero_segment_id)
        ' the CRF step '
        prev_transpose = brew.transpose(step_model, cell_t_prev, [s('prev_transpose')], axes=(0, 2, 1))
        prev_tiled = step_model.net.Tile(prev_transpose, [s('prev_tiled')], tiles=self.num_classes_padded, axis=2)
        input_t_tiled = step_model.net.Tile(input_t, [s('input_t_tiled')], tiles=self.num_classes_padded, axis=1)
        input_with_prev = step_model.net.Add([prev_tiled, input_t_tiled], [s('input_with_prev')])
        all_with_transitions = step_model.net.Add([input_with_prev, transitions], [s('prev_with_transitions')], broadcast=1, use_grad_hack=1)
        (all_with_transitions_reshaped, _) = step_model.net.Reshape(all_with_transitions, [s('all_with_transitions_reshaped'), s('all_with_transitions_orig')], shape=(self.num_classes_padded, self.num_classes_padded))
        cell_t = step_model.net.SortedSegmentRangeLogSumExp([all_with_transitions_reshaped, zero_segment_id], [s('cell_t')])
        step_model.net.AddExternalOutputs(cell_t)
        ' recurrent network '
        cell_input_blob = initial_state
        (out_all, out_last) = recurrent.recurrent_net(net=self.model.net, cell_net=step_model.net, inputs=[(input_t, input_blob)], initial_cell_inputs=[(cell_t_prev, cell_input_blob)], links={cell_t_prev: cell_t}, scope=scope, outputs_with_grads=(1,))
        return out_last

    def update_predictions(self, classes):
        if False:
            i = 10
            return i + 15

        def crf_update_predictions_op(inputs, outputs):
            if False:
                i = 10
                return i + 15
            predictions = inputs[0].data
            transitions = inputs[1].data
            predictions = inputs[0].data
            predictions_shape = inputs[0].shape
            outputs[0].reshape(predictions_shape)
            trellis = np.zeros(predictions_shape)
            backpointers = np.zeros(predictions_shape, dtype=np.int32)
            trellis[0] = predictions[0]
            for t in range(1, predictions_shape[0]):
                v = np.expand_dims(trellis[t - 1], 1) + transitions
                trellis[t] = predictions[t] + np.max(v, 0)
                backpointers[t] = np.argmax(v, 0)
            viterbi = [np.argmax(trellis[-1])]
            for bp in reversed(backpointers[1:]):
                viterbi.append(bp[viterbi[-1]])
            viterbi.reverse()
            new_predictions = np.zeros(predictions_shape)
            old_bests = []
            for (i, w_predictions) in enumerate(predictions):
                new_predictions[i] = predictions[i]
                old_best = np.argmax(w_predictions)
                old_bests.append(old_best)
                (w_predictions[viterbi[i]], w_predictions[old_best]) = (w_predictions[old_best], w_predictions[viterbi[i]])
                new_predictions[i] = w_predictions
            orig_predictions = new_predictions[1:-1, 0:-2]
            outputs[0].reshape(orig_predictions.shape)
            outputs[0].data[...] = orig_predictions
        padded_classes = CRFWithLoss.pad_predictions(classes, self.model.param_init_net, self.model.net, self.num_classes)
        new_classes = self.model.net.Python(crf_update_predictions_op)([padded_classes, self.transitions], core.ScopedBlobReference('post_crf_classes'))
        return new_classes

    @staticmethod
    def pad_labels(labels, init_net, net, num_classes):
        if False:
            print('Hello World!')
        bos_i = num_classes
        eos_i = num_classes + 1
        bos_i_b = init_net.ConstantFill([], shape=[1], value=bos_i)
        eos_i_b = init_net.ConstantFill([], shape=[1], value=eos_i)
        labels = net.Cast([labels], to='int64')
        (padded_labels, _) = net.Concat([bos_i_b, labels, eos_i_b], axis=0, outputs=2)
        return padded_labels

    @staticmethod
    def pad_predictions(predictions, init_net, net, num_classes):
        if False:
            return 10
        low_score = -1000.0
        b_scores = np.array([[low_score] * num_classes + [0, low_score]]).astype(np.float32)
        e_scores = np.array([[low_score] * num_classes + [low_score, 0]]).astype(np.float32)
        b_scores = init_net.GivenTensorFill([], 'b_scores', shape=[1, num_classes + 2], values=b_scores)
        e_scores = init_net.GivenTensorFill([], 'e_scores', shape=[1, num_classes + 2], values=e_scores)
        zero_index = net.ConstantFill([], shape=[1], value=0)
        length = net.Gather([net.Shape([predictions]), zero_index])
        length = net.Cast(length, to='int32')
        t_range = net.LengthsRangeFill(length)
        padding = net.ConstantFill([t_range], value=low_score)
        padding = net.ExpandDims(padding, dims=[1])
        (padded_predictions, _) = net.Concat([predictions, padding, padding], outputs=2, axis=1)
        (padded_predictions_concat, _) = net.Concat([b_scores, padded_predictions, e_scores], outputs=2, axis=0)
        return padded_predictions_concat