import chainer
from chainer.functions.activation import sigmoid
from chainer.functions.activation import tanh
from chainer.functions.array import reshape
from chainer.functions.array import split_axis
from chainer.functions.noise import zoneout
from chainer import link
from chainer.links.connection import linear
from chainer.utils import argument
from chainer import variable

class StatefulZoneoutLSTM(link.Chain):

    def __init__(self, in_size, out_size, c_ratio=0.5, h_ratio=0.5, **kwargs):
        if False:
            while True:
                i = 10
        if kwargs:
            argument.check_unexpected_kwargs(kwargs, train='train argument is not supported anymore. Use chainer.using_config')
            argument.assert_kwargs_empty(kwargs)
        super(StatefulZoneoutLSTM, self).__init__()
        self.state_size = out_size
        self.c_ratio = c_ratio
        self.h_ratio = h_ratio
        self.reset_state()
        with self.init_scope():
            self.upward = linear.Linear(in_size, 4 * out_size)
            self.lateral = linear.Linear(out_size, 4 * out_size, nobias=True)

    def device_resident_accept(self, visitor):
        if False:
            return 10
        super(StatefulZoneoutLSTM, self).device_resident_accept(visitor)
        if self.c is not None:
            visitor.visit_variable(self.c)
        if self.h is not None:
            visitor.visit_variable(self.h)

    def set_state(self, c, h):
        if False:
            i = 10
            return i + 15
        'Sets the internal state.\n\n        It sets the :attr:`c` and :attr:`h` attributes.\n\n        Args:\n            c (~chainer.Variable): A new cell states of LSTM units.\n            h (~chainer.Variable): A new output at the previous time step.\n\n        '
        assert isinstance(c, variable.Variable)
        assert isinstance(h, variable.Variable)
        c.to_device(self.device)
        h.to_device(self.device)
        self.c = c
        self.h = h

    def reset_state(self):
        if False:
            while True:
                i = 10
        'Resets the internal state.\n\n        It sets ``None`` to the :attr:`c` and :attr:`h` attributes.\n\n        '
        self.c = self.h = None

    def forward(self, x):
        if False:
            for i in range(10):
                print('nop')
        'Updates the internal state and returns the LSTM outputs.\n\n        Args:\n            x (~chainer.Variable): A new batch from the input sequence.\n\n        Returns:\n            ~chainer.Variable: Outputs of updated LSTM units.\n\n        '
        lstm_in = self.upward(x)
        if self.h is not None:
            lstm_in += self.lateral(self.h)
        else:
            xp = self.xp
            with chainer.using_device(self.device):
                self.h = variable.Variable(xp.zeros((len(x), self.state_size), dtype=x.dtype))
        if self.c is None:
            xp = self.xp
            with chainer.using_device(self.device):
                self.c = variable.Variable(xp.zeros((len(x), self.state_size), dtype=x.dtype))
        lstm_in = reshape.reshape(lstm_in, (len(lstm_in), lstm_in.shape[1] // 4, 4))
        (a, i, f, o) = split_axis.split_axis(lstm_in, 4, 2)
        a = reshape.reshape(a, (len(a), self.state_size))
        i = reshape.reshape(i, (len(i), self.state_size))
        f = reshape.reshape(f, (len(f), self.state_size))
        o = reshape.reshape(o, (len(o), self.state_size))
        c_tmp = tanh.tanh(a) * sigmoid.sigmoid(i) + sigmoid.sigmoid(f) * self.c
        self.c = zoneout.zoneout(self.c, c_tmp, self.c_ratio)
        self.h = zoneout.zoneout(self.h, sigmoid.sigmoid(o) * tanh.tanh(c_tmp), self.h_ratio)
        return self.h