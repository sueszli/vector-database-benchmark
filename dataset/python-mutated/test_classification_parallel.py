import pycaret.classification as pc
from pycaret.datasets import get_data

def _score_dummy(y_true, y_prob, axis=0):
    if False:
        print('Hello World!')
    return 0.0

def test_classification_parallel():
    if False:
        while True:
            i = 10
    from pycaret.parallel import FugueBackend
    pc.setup(data_func=lambda : get_data('juice', verbose=False), target='Purchase', session_id=0, n_jobs=1, verbose=False, html=False)
    test_models = pc.models().index.tolist()[:5]
    pc.compare_models(include=test_models, n_select=2)
    pc.compare_models(include=test_models, n_select=2, parallel=FugueBackend('dask'))
    fconf = {'fugue.rpc.server': 'fugue.rpc.flask.FlaskRPCServer', 'fugue.rpc.flask_server.host': 'localhost', 'fugue.rpc.flask_server.port': '3333', 'fugue.rpc.flask_server.timeout': '2 sec'}
    be = FugueBackend('dask', fconf, display_remote=True, batch_size=3, top_only=False)
    pc.compare_models(n_select=2, parallel=be)
    res = pc.pull()
    assert res.shape[0] > 10
    pc.add_metric(id='mydummy', name='DUMMY', score_func=_score_dummy, target='pred_proba', greater_is_better=True)
    pc.compare_models(n_select=2, sort='DUMMY', parallel=be)
    pc.pull()