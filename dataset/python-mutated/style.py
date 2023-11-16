from pygments.token import Token
from pygments.util import ClassNotFound
from prompt_toolkit.styles import default_style_extensions, style_from_dict
import pygments.styles

class StyleFactory(object):
    """Provide styles for the autocomplete menu and the toolbar.

    :type style: :class:`pygments.style.StyleMeta`
    :param style: An instance of `pygments.style.StyleMeta`.
    """

    def __init__(self, name):
        if False:
            while True:
                i = 10
        self.style = self.style_factory(name)

    def style_factory(self, name):
        if False:
            return 10
        'Retrieve the specified pygments style.\n\n        If the specified style is not found, the vim style is returned.\n\n        :type style_name: str\n        :param style_name: The pygments style name.\n\n        :rtype: :class:`pygments.style.StyleMeta`\n        :return: An instance of `pygments.style.StyleMeta`.\n        '
        try:
            style = pygments.styles.get_style_by_name(name)
        except ClassNotFound:
            style = pygments.styles.get_style_by_name('native')
        styles = {}
        styles.update(style.styles)
        styles.update(default_style_extensions)
        styles.update({Token.Menu.Completions.Completion.Current: 'bg:#00aaaa #000000', Token.Menu.Completions.Completion: 'bg:#008888 #ffffff', Token.Menu.Completions.Meta.Current: 'bg:#00aaaa #000000', Token.Menu.Completions.Meta: 'bg:#00aaaa #ffffff', Token.Menu.Completions.ProgressButton: 'bg:#003333', Token.Menu.Completions.ProgressBar: 'bg:#00aaaa', Token.Scrollbar: 'bg:#00aaaa', Token.Scrollbar.Button: 'bg:#003333', Token.Toolbar: 'bg:#222222 #cccccc', Token.Toolbar.Off: 'bg:#222222 #696969', Token.Toolbar.On: 'bg:#222222 #ffffff', Token.Toolbar.Search: 'noinherit bold', Token.Toolbar.Search.Text: 'nobold', Token.Toolbar.System: 'noinherit bold', Token.Toolbar.Arg: 'noinherit bold', Token.Toolbar.Arg.Text: 'nobold'})
        return style_from_dict(styles)