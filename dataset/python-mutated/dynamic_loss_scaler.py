class DynamicLossScaler(object):

    def __init__(self, init_scale=2.0 ** 15, scale_factor=2.0, scale_window=2000, tolerance=0.0, threshold=None, min_loss_scale=0.0001):
        if False:
            while True:
                i = 10
        self.loss_scale = init_scale
        self.scale_factor = scale_factor
        self.scale_window = scale_window
        self.tolerance = tolerance
        self.threshold = threshold
        self._iter = 0
        self._last_overflow_iter = -1
        self._last_rescale_iter = -1
        self._overflows_since_rescale = 0
        self.min_loss_scale = min_loss_scale

    def scale(self, outputs):
        if False:
            for i in range(10):
                print('nop')
        return self.loss_scale * outputs

    def update(self):
        if False:
            for i in range(10):
                print('nop')
        if (self._iter - self._last_overflow_iter) % self.scale_window == 0:
            self.loss_scale *= self.scale_factor
            self._last_rescale_iter = self._iter
        self._iter += 1

    def _decrease_loss_scale(self):
        if False:
            return 10
        self.loss_scale /= self.scale_factor
        if self.threshold is not None:
            self.loss_scale = max(self.loss_scale, self.threshold)

    def check_overflow(self, grad_norm):
        if False:
            return 10
        if grad_norm == float('inf') or grad_norm != grad_norm:
            prev_scale = self.loss_scale
            iter_since_rescale = self._iter - self._last_rescale_iter
            self._last_overflow_iter = self._iter
            self._overflows_since_rescale += 1
            pct_overflow = self._overflows_since_rescale / float(iter_since_rescale)
            if pct_overflow >= self.tolerance:
                self._decrease_loss_scale()
                self._last_rescale_iter = self._iter
                self._overflows_since_rescale = 0
            if self.loss_scale <= self.min_loss_scale:
                self.loss_scale = prev_scale
                raise FloatingPointError('Minimum loss scale reached ({}). Your loss is probably exploding. Try lowering the learning rate, using gradient clipping or increasing the batch size.'.format(self.min_loss_scale))
            self._iter += 1
            raise OverflowError('setting loss scale to: ' + str(self.loss_scale))