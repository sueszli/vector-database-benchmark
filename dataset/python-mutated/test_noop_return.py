import logging
log = logging.getLogger(__name__)

def test_noop_return(caplog, salt_cli, salt_minion):
    if False:
        for i in range(10):
            print('nop')
    with caplog.at_level(logging.DEBUG):
        salt_cli.run('test.ping', minion_tgt=salt_minion.id)
        assert 'NOOP_RETURN' in caplog.text