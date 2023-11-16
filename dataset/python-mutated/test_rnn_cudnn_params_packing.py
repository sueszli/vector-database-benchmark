from unittest import TestCase
import paddle

def create_model():
    if False:
        return 10
    hidden_size = 32
    bilstm = paddle.nn.LSTM(hidden_size, hidden_size, num_layers=1, direction='bidirectional')
    return bilstm

class TestRNNProgramClone(TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        paddle.enable_static()

    def test_rnn_with_cudnn_clone(self):
        if False:
            i = 10
            return i + 15
        train_program = paddle.static.Program()
        test_program = paddle.static.Program()
        startup_prog = paddle.static.Program()
        with paddle.static.program_guard(train_program, startup_prog):
            with paddle.base.unique_name.guard():
                bilstm = create_model()
        with paddle.base.program_guard(test_program, startup_prog):
            with paddle.base.unique_name.guard():
                bilstm = create_model()