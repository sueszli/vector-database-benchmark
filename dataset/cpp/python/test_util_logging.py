"""Test logging util."""

import codecs
import os
import os.path

import pytest
from docutils import nodes

from sphinx.errors import SphinxWarning
from sphinx.testing.util import strip_escseq
from sphinx.util import logging, osutil
from sphinx.util.console import colorize
from sphinx.util.logging import is_suppressed_warning, prefixed_warnings
from sphinx.util.parallel import ParallelTasks


def test_info_and_warning(app, status, warning):
    app.verbosity = 2
    logging.setup(app, status, warning)
    logger = logging.getLogger(__name__)

    logger.debug('message1')
    logger.info('message2')
    logger.warning('message3')
    logger.critical('message4')
    logger.error('message5')

    assert 'message1' in status.getvalue()
    assert 'message2' in status.getvalue()
    assert 'message3' not in status.getvalue()
    assert 'message4' not in status.getvalue()
    assert 'message5' not in status.getvalue()

    assert 'message1' not in warning.getvalue()
    assert 'message2' not in warning.getvalue()
    assert 'WARNING: message3' in warning.getvalue()
    assert 'CRITICAL: message4' in warning.getvalue()
    assert 'ERROR: message5' in warning.getvalue()


def test_Exception(app, status, warning):
    logging.setup(app, status, warning)
    logger = logging.getLogger(__name__)

    logger.info(Exception)
    assert "<class 'Exception'>" in status.getvalue()


def test_verbosity_filter(app, status, warning):
    # verbosity = 0: INFO
    app.verbosity = 0
    logging.setup(app, status, warning)
    logger = logging.getLogger(__name__)

    logger.info('message1')
    logger.verbose('message2')
    logger.debug('message3')

    assert 'message1' in status.getvalue()
    assert 'message2' not in status.getvalue()
    assert 'message3' not in status.getvalue()
    assert 'message4' not in status.getvalue()

    # verbosity = 1: VERBOSE
    app.verbosity = 1
    logging.setup(app, status, warning)
    logger = logging.getLogger(__name__)

    logger.info('message1')
    logger.verbose('message2')
    logger.debug('message3')

    assert 'message1' in status.getvalue()
    assert 'message2' in status.getvalue()
    assert 'message3' not in status.getvalue()
    assert 'message4' not in status.getvalue()

    # verbosity = 2: DEBUG
    app.verbosity = 2
    logging.setup(app, status, warning)
    logger = logging.getLogger(__name__)

    logger.info('message1')
    logger.verbose('message2')
    logger.debug('message3')

    assert 'message1' in status.getvalue()
    assert 'message2' in status.getvalue()
    assert 'message3' in status.getvalue()
    assert 'message4' not in status.getvalue()


def test_nonl_info_log(app, status, warning):
    logging.setup(app, status, warning)
    logger = logging.getLogger(__name__)

    logger.info('message1', nonl=True)
    logger.info('message2')
    logger.info('message3')

    assert 'message1message2\nmessage3' in status.getvalue()


def test_once_warning_log(app, status, warning):
    logging.setup(app, status, warning)
    logger = logging.getLogger(__name__)

    logger.warning('message: %d', 1, once=True)
    logger.warning('message: %d', 1, once=True)
    logger.warning('message: %d', 2, once=True)

    assert 'WARNING: message: 1\nWARNING: message: 2\n' in strip_escseq(warning.getvalue())


def test_is_suppressed_warning():
    suppress_warnings = ["ref", "files.*", "rest.duplicated_labels"]

    assert is_suppressed_warning(None, None, suppress_warnings) is False
    assert is_suppressed_warning("ref", None, suppress_warnings) is True
    assert is_suppressed_warning("ref", "numref", suppress_warnings) is True
    assert is_suppressed_warning("ref", "option", suppress_warnings) is True
    assert is_suppressed_warning("files", "image", suppress_warnings) is True
    assert is_suppressed_warning("files", "stylesheet", suppress_warnings) is True
    assert is_suppressed_warning("rest", None, suppress_warnings) is False
    assert is_suppressed_warning("rest", "syntax", suppress_warnings) is False
    assert is_suppressed_warning("rest", "duplicated_labels", suppress_warnings) is True


def test_suppress_warnings(app, status, warning):
    logging.setup(app, status, warning)
    logger = logging.getLogger(__name__)

    app._warncount = 0  # force reset

    app.config.suppress_warnings = []
    warning.truncate(0)
    logger.warning('message0', type='test')
    logger.warning('message1', type='test', subtype='logging')
    logger.warning('message2', type='test', subtype='crash')
    logger.warning('message3', type='actual', subtype='logging')
    assert 'message0' in warning.getvalue()
    assert 'message1' in warning.getvalue()
    assert 'message2' in warning.getvalue()
    assert 'message3' in warning.getvalue()
    assert app._warncount == 4

    app.config.suppress_warnings = ['test']
    warning.truncate(0)
    logger.warning('message0', type='test')
    logger.warning('message1', type='test', subtype='logging')
    logger.warning('message2', type='test', subtype='crash')
    logger.warning('message3', type='actual', subtype='logging')
    assert 'message0' not in warning.getvalue()
    assert 'message1' not in warning.getvalue()
    assert 'message2' not in warning.getvalue()
    assert 'message3' in warning.getvalue()
    assert app._warncount == 5

    app.config.suppress_warnings = ['test.logging']
    warning.truncate(0)
    logger.warning('message0', type='test')
    logger.warning('message1', type='test', subtype='logging')
    logger.warning('message2', type='test', subtype='crash')
    logger.warning('message3', type='actual', subtype='logging')
    assert 'message0' in warning.getvalue()
    assert 'message1' not in warning.getvalue()
    assert 'message2' in warning.getvalue()
    assert 'message3' in warning.getvalue()
    assert app._warncount == 8


def test_warningiserror(app, status, warning):
    logging.setup(app, status, warning)
    logger = logging.getLogger(__name__)

    # if False, warning is not error
    app.warningiserror = False
    logger.warning('message')

    # if True, warning raises SphinxWarning exception
    app.warningiserror = True
    with pytest.raises(SphinxWarning):
        logger.warning('message: %s', 'arg')

    # message contains format string (refs: #4070)
    with pytest.raises(SphinxWarning):
        logger.warning('%s')


def test_info_location(app, status, warning):
    logging.setup(app, status, warning)
    logger = logging.getLogger(__name__)

    logger.info('message1', location='index')
    assert 'index.txt: message1' in status.getvalue()

    logger.info('message2', location=('index', 10))
    assert 'index.txt:10: message2' in status.getvalue()

    logger.info('message3', location=None)
    assert '\nmessage3' in status.getvalue()

    node = nodes.Node()
    node.source, node.line = ('index.txt', 10)
    logger.info('message4', location=node)
    assert 'index.txt:10: message4' in status.getvalue()

    node.source, node.line = ('index.txt', None)
    logger.info('message5', location=node)
    assert 'index.txt:: message5' in status.getvalue()

    node.source, node.line = (None, 10)
    logger.info('message6', location=node)
    assert '<unknown>:10: message6' in status.getvalue()

    node.source, node.line = (None, None)
    logger.info('message7', location=node)
    assert '\nmessage7' in status.getvalue()


def test_warning_location(app, status, warning):
    logging.setup(app, status, warning)
    logger = logging.getLogger(__name__)

    logger.warning('message1', location='index')
    assert 'index.txt: WARNING: message1' in warning.getvalue()

    logger.warning('message2', location=('index', 10))
    assert 'index.txt:10: WARNING: message2' in warning.getvalue()

    logger.warning('message3', location=None)
    assert colorize('red', 'WARNING: message3') in warning.getvalue()

    node = nodes.Node()
    node.source, node.line = ('index.txt', 10)
    logger.warning('message4', location=node)
    assert 'index.txt:10: WARNING: message4' in warning.getvalue()

    node.source, node.line = ('index.txt', None)
    logger.warning('message5', location=node)
    assert 'index.txt:: WARNING: message5' in warning.getvalue()

    node.source, node.line = (None, 10)
    logger.warning('message6', location=node)
    assert '<unknown>:10: WARNING: message6' in warning.getvalue()

    node.source, node.line = (None, None)
    logger.warning('message7', location=node)
    assert colorize('red', 'WARNING: message7') in warning.getvalue()


def test_suppress_logging(app, status, warning):
    logging.setup(app, status, warning)
    logger = logging.getLogger(__name__)

    logger.warning('message1')
    with logging.suppress_logging():
        logger.warning('message2')
        assert 'WARNING: message1' in warning.getvalue()
        assert 'WARNING: message2' not in warning.getvalue()

    assert 'WARNING: message1' in warning.getvalue()
    assert 'WARNING: message2' not in warning.getvalue()


def test_pending_warnings(app, status, warning):
    logging.setup(app, status, warning)
    logger = logging.getLogger(__name__)

    logger.warning('message1')
    with logging.pending_warnings():
        # not logged yet (buffered) in here
        logger.warning('message2')
        logger.warning('message3')
        assert 'WARNING: message1' in warning.getvalue()
        assert 'WARNING: message2' not in warning.getvalue()
        assert 'WARNING: message3' not in warning.getvalue()

    # actually logged as ordered
    assert 'WARNING: message2\nWARNING: message3' in strip_escseq(warning.getvalue())


def test_colored_logs(app, status, warning):
    app.verbosity = 2
    logging.setup(app, status, warning)
    logger = logging.getLogger(__name__)

    # default colors
    logger.debug('message1')
    logger.verbose('message2')
    logger.info('message3')
    logger.warning('message4')
    logger.critical('message5')
    logger.error('message6')

    assert colorize('darkgray', 'message1') in status.getvalue()
    assert 'message2\n' in status.getvalue()  # not colored
    assert 'message3\n' in status.getvalue()  # not colored
    assert colorize('red', 'WARNING: message4') in warning.getvalue()
    assert 'CRITICAL: message5\n' in warning.getvalue()  # not colored
    assert colorize('darkred', 'ERROR: message6') in warning.getvalue()

    # color specification
    logger.debug('message7', color='white')
    logger.info('message8', color='red')
    assert colorize('white', 'message7') in status.getvalue()
    assert colorize('red', 'message8') in status.getvalue()


@pytest.mark.xfail(os.name != 'posix',
                   reason="Parallel mode does not work on Windows")
def test_logging_in_ParallelTasks(app, status, warning):
    logging.setup(app, status, warning)
    logger = logging.getLogger(__name__)

    def child_process():
        logger.info('message1')
        logger.warning('message2', location='index')

    tasks = ParallelTasks(1)
    tasks.add_task(child_process)
    tasks.join()
    assert 'message1' in status.getvalue()
    assert 'index.txt: WARNING: message2' in warning.getvalue()


def test_output_with_unencodable_char(app, status, warning):
    class StreamWriter(codecs.StreamWriter):
        def write(self, object):
            self.stream.write(object.encode('cp1252').decode('cp1252'))

    logging.setup(app, StreamWriter(status), warning)
    logger = logging.getLogger(__name__)

    # info with UnicodeEncodeError
    status.truncate(0)
    status.seek(0)
    logger.info("unicode \u206d...")
    assert status.getvalue() == "unicode ?...\n"


def test_skip_warningiserror(app, status, warning):
    logging.setup(app, status, warning)
    logger = logging.getLogger(__name__)

    app.warningiserror = True
    with logging.skip_warningiserror():
        logger.warning('message')

    # if False, warning raises SphinxWarning exception
    with logging.skip_warningiserror(False):  # NoQA: SIM117
        with pytest.raises(SphinxWarning):
            logger.warning('message')

    # It also works during pending_warnings.
    with logging.pending_warnings():  # NoQA: SIM117
        with logging.skip_warningiserror():
            logger.warning('message')

    with pytest.raises(SphinxWarning):  # NoQA: PT012,SIM117
        with logging.pending_warnings():
            with logging.skip_warningiserror(False):
                logger.warning('message')


def test_prefixed_warnings(app, status, warning):
    logging.setup(app, status, warning)
    logger = logging.getLogger(__name__)

    logger.warning('message1')
    with prefixed_warnings('PREFIX:'):
        logger.warning('message2')
        with prefixed_warnings('Another PREFIX:'):
            logger.warning('message3')
        logger.warning('message4')
    logger.warning('message5')

    assert 'WARNING: message1' in warning.getvalue()
    assert 'WARNING: PREFIX: message2' in warning.getvalue()
    assert 'WARNING: Another PREFIX: message3' in warning.getvalue()
    assert 'WARNING: PREFIX: message4' in warning.getvalue()
    assert 'WARNING: message5' in warning.getvalue()


def test_get_node_location_abspath():
    # Ensure that node locations are reported as an absolute path,
    # even if the source attribute is a relative path.

    relative_filename = os.path.join('relative', 'path.txt')
    absolute_filename = osutil.abspath(relative_filename)

    n = nodes.Node()
    n.source = relative_filename

    location = logging.get_node_location(n)

    assert location == absolute_filename + ':'
