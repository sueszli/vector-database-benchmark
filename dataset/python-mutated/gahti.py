from volatility import renderers
from volatility.renderers.basic import Address
from volatility.renderers.text import TextRenderer
import volatility.utils as utils
import volatility.debug as debug
import volatility.plugins.gui.constants as consts
import volatility.plugins.gui.sessions as sessions

class Gahti(sessions.Sessions):
    """Dump the USER handle type information"""

    def unified_output(self, data):
        if False:
            while True:
                i = 10
        return renderers.TreeGrid([('Session', str), ('Type', str), ('Tag', str), ('fnDestroy', Address), ('Flags', str)], self.generator(data))

    def generator(self, data):
        if False:
            print('Hello World!')
        profile = utils.load_as(self._config).profile
        version = (profile.metadata.get('major', 0), profile.metadata.get('minor', 0))
        if version >= (6, 1):
            handle_types = consts.HANDLE_TYPE_ENUM_SEVEN
        else:
            handle_types = consts.HANDLE_TYPE_ENUM
        for session in data:
            gahti = session.find_gahti()
            if gahti:
                for (i, h) in handle_types.items():
                    yield (0, [str(session.SessionId), str(h), str(gahti.types[i].dwAllocTag), Address(gahti.types[i].fnDestroy), str(gahti.types[i].bObjectCreateFlags)])

    def render_text(self, outfd, data):
        if False:
            return 10
        output = self.unified_output(data)
        if isinstance(output, renderers.TreeGrid):
            tr = TextRenderer(self.text_cell_renderers, sort_column=self.text_sort_column)
            tr.render(outfd, output)
        else:
            raise TypeError('Unified Output must return a TreeGrid object')