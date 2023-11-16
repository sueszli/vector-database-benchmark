import logging
from salt._logging import DFLT_LOG_FMT_JID
from tests.support.helpers import PRE_PYTEST_SKIP

@PRE_PYTEST_SKIP
def test_jid_in_logs(caplog, salt_call_cli):
    if False:
        while True:
            i = 10
    '\n    Test JID in log_format\n    '
    jid_formatted_str = DFLT_LOG_FMT_JID.split('%')[0]
    formatter = logging.Formatter(fmt='%(jid)s %(message)s')
    with caplog.at_level(logging.DEBUG):
        previous_formatter = caplog.handler.formatter
        try:
            caplog.handler.setFormatter(formatter)
            ret = salt_call_cli.run('test.ping')
            assert ret.returncode == 0
            assert ret.data is True
            assert_error_msg = "'{}' not found in log messages:\n>>>>>>>>>{}\n<<<<<<<<<".format(jid_formatted_str, caplog.text)
            assert jid_formatted_str in caplog.text, assert_error_msg
        finally:
            caplog.handler.setFormatter(previous_formatter)