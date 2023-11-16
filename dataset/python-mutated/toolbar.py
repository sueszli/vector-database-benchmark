from __future__ import print_function
from pygments.token import Token

class Toolbar(object):
    """Show information about the aws-shell in a tool bar.

    :type handler: callable
    :param handler: Wraps the callable `get_toolbar_items`.
    """

    def __init__(self, paginate_comments_cfg):
        if False:
            return 10
        self.handler = self._create_toolbar_handler(paginate_comments_cfg)

    def _create_toolbar_handler(self, paginate_comments_cfg):
        if False:
            i = 10
            return i + 15
        'Create the toolbar handler.\n\n        :type paginate_comments_cfg: callable\n        :param paginate_comments_cfg: Specifies whether to paginate comments.\n\n        :rtype: callable\n        :returns: get_toolbar_items.\n        '
        assert callable(paginate_comments_cfg)

        def get_toolbar_items(_):
            if False:
                return 10
            'Return the toolbar items.\n\n            :type _: :class:`prompt_toolkit.Cli`\n            :param _: (Unused)\n\n            :rtype: list\n            :return: A list of (pygments.Token.Toolbar, str).\n            '
            return [(Token.Toolbar, ' [F10] Exit ')]
        return get_toolbar_items