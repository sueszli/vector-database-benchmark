from enum import Enum
from docutils.nodes import Element
from sphinx.domains.python import PyXRefRole
from sphinx.environment import BuildEnvironment
from sphinx.util import logging
import telegram
sphinx_logger = logging.getLogger(__name__)
CONSTANTS_ROLE = 'tg-const'

class TGConstXRefRole(PyXRefRole):
    """This is a bit of Sphinx magic. We add a new role type called tg-const that allows us to
    reference values from the `telegram.constants.module` while using the actual value as title
    of the link.

    Example:

        :tg-const:`telegram.constants.MessageLimit.MAX_TEXT_LENGTH` renders as `4096` but links to
        the constant.
    """

    def process_link(self, env: BuildEnvironment, refnode: Element, has_explicit_title: bool, title: str, target: str) -> tuple[str, str]:
        if False:
            i = 10
            return i + 15
        (title, target) = super().process_link(env, refnode, has_explicit_title, title, target)
        try:
            value = eval(target)
            if isinstance(value, Enum):
                if isinstance(value, telegram.constants.FileSizeLimit):
                    return (f'{int(value.value / 1000000.0)} MB', target)
                return (repr(value.value), target)
            if isinstance(value, str) and target in ('telegram.constants.BOT_API_VERSION', 'telegram.__version__'):
                return (value, target)
            if isinstance(value, tuple) and target in ('telegram.constants.BOT_API_VERSION_INFO', 'telegram.__version_info__'):
                return (repr(value), target)
            sphinx_logger.warning(f'%s:%d: WARNING: Did not convert reference %s. :{CONSTANTS_ROLE}: is not supposed to be used with this type of target.', refnode.source, refnode.line, refnode.rawsource)
            return (title, target)
        except Exception as exc:
            sphinx_logger.exception('%s:%d: WARNING: Did not convert reference %s due to an exception.', refnode.source, refnode.line, refnode.rawsource, exc_info=exc)
            return (title, target)