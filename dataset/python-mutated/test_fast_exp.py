from ..fast_exp import fast_exp
import numpy as np

def test_fast_exp():
    if False:
        return 10
    X = np.linspace(-5, 0, 5000, endpoint=True)
    Y = np.exp(X)
    _y_f64 = np.array([fast_exp['float64_t'](x) for x in X])
    _y_f32 = np.array([fast_exp['float32_t'](x) for x in X.astype('float32')], dtype='float32')
    for _y in [_y_f64, _y_f32]:
        assert np.abs(Y - _y).mean() < 0.003