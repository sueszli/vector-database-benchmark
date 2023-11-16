from __future__ import annotations
import pytest
ARGS = dict(foo=False, bar=[1, 2, 3], bam='bam', baz=u'baz')
ARGUMENT_SPEC = dict(foo=dict(default=True, type='bool'), bar=dict(default=[], type='list'), bam=dict(default='bam'), baz=dict(default=u'baz'), password=dict(default=True), no_log=dict(default="you shouldn't see me", no_log=True))

@pytest.mark.parametrize('am, stdin', [(ARGUMENT_SPEC, ARGS)], indirect=['am', 'stdin'])
def test_module_utils_basic__log_invocation(am, mocker):
    if False:
        return 10
    am.log = mocker.MagicMock()
    am._log_invocation()
    args = am.log.call_args[0]
    assert len(args) == 1
    message = args[0]
    assert len(message) == len('Invoked with bam=bam bar=[1, 2, 3] foo=False baz=baz no_log=NOT_LOGGING_PARAMETER password=NOT_LOGGING_PASSWORD')
    assert message.startswith('Invoked with ')
    assert ' bam=bam' in message
    assert ' bar=[1, 2, 3]' in message
    assert ' foo=False' in message
    assert ' baz=baz' in message
    assert ' no_log=NOT_LOGGING_PARAMETER' in message
    assert ' password=NOT_LOGGING_PASSWORD' in message
    kwargs = am.log.call_args[1]
    assert kwargs == dict(log_args={'foo': 'False', 'bar': '[1, 2, 3]', 'bam': 'bam', 'baz': 'baz', 'password': 'NOT_LOGGING_PASSWORD', 'no_log': 'NOT_LOGGING_PARAMETER'})