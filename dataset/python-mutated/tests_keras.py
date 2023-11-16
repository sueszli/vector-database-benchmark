from .tests_tqdm import importorskip, mark
pytestmark = mark.slow

@mark.filterwarnings('ignore:.*:DeprecationWarning')
def test_keras(capsys):
    if False:
        print('Hello World!')
    'Test tqdm.keras.TqdmCallback'
    TqdmCallback = importorskip('tqdm.keras').TqdmCallback
    np = importorskip('numpy')
    try:
        import keras as K
    except ImportError:
        K = importorskip('tensorflow.keras')
    dtype = np.float32
    model = K.models.Sequential([K.layers.InputLayer((1, 1), dtype=dtype), K.layers.Conv1D(1, 1)])
    model.compile('adam', 'mse')
    x = np.random.rand(100, 1, 1).astype(dtype)
    batch_size = 10
    batches = len(x) / batch_size
    epochs = 5
    model.fit(x, x, epochs=epochs, batch_size=batch_size, verbose=False, callbacks=[TqdmCallback(epochs, desc='training', data_size=len(x), batch_size=batch_size, verbose=0)])
    (_, res) = capsys.readouterr()
    assert 'training: ' in res
    assert '{epochs}/{epochs}'.format(epochs=epochs) in res
    assert '{batches}/{batches}'.format(batches=batches) not in res
    model.fit(x, x, epochs=epochs, batch_size=batch_size, verbose=False, callbacks=[TqdmCallback(epochs, desc='training', data_size=len(x), batch_size=batch_size, verbose=2)])
    (_, res) = capsys.readouterr()
    assert 'training: ' in res
    assert '{epochs}/{epochs}'.format(epochs=epochs) in res
    assert '{batches}/{batches}'.format(batches=batches) in res
    model.fit(x, x, epochs=epochs, batch_size=batch_size, verbose=False, callbacks=[TqdmCallback(desc='training', verbose=2)])
    (_, res) = capsys.readouterr()
    assert 'training: ' in res
    assert '{epochs}/{epochs}'.format(epochs=epochs) in res
    assert '{batches}/{batches}'.format(batches=batches) in res
    initial_epoch = 3
    model.fit(x, x, initial_epoch=initial_epoch, epochs=epochs, batch_size=batch_size, verbose=False, callbacks=[TqdmCallback(desc='training', verbose=0, miniters=1, mininterval=0, maxinterval=0)])
    (_, res) = capsys.readouterr()
    assert 'training: ' in res
    assert '{epochs}/{epochs}'.format(epochs=initial_epoch - 1) not in res
    assert '{epochs}/{epochs}'.format(epochs=epochs) in res