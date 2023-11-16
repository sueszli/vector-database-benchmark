"""Interactive figures in the IPython notebook."""
from base64 import b64encode
import io
import json
import pathlib
import uuid
from ipykernel.comm import Comm
from IPython.display import display, Javascript, HTML
from matplotlib import is_interactive
from matplotlib._pylab_helpers import Gcf
from matplotlib.backend_bases import _Backend, CloseEvent, NavigationToolbar2
from .backend_webagg_core import FigureCanvasWebAggCore, FigureManagerWebAgg, NavigationToolbar2WebAgg
from .backend_webagg_core import TimerTornado, TimerAsyncio

def connection_info():
    if False:
        print('Hello World!')
    '\n    Return a string showing the figure and connection status for the backend.\n\n    This is intended as a diagnostic tool, and not for general use.\n    '
    result = ['{fig} - {socket}'.format(fig=manager.canvas.figure.get_label() or f'Figure {manager.num}', socket=manager.web_sockets) for manager in Gcf.get_all_fig_managers()]
    if not is_interactive():
        result.append(f'Figures pending show: {len(Gcf.figs)}')
    return '\n'.join(result)
_FONT_AWESOME_CLASSES = {'home': 'fa fa-home', 'back': 'fa fa-arrow-left', 'forward': 'fa fa-arrow-right', 'zoom_to_rect': 'fa fa-square-o', 'move': 'fa fa-arrows', 'download': 'fa fa-floppy-o', None: None}

class NavigationIPy(NavigationToolbar2WebAgg):
    toolitems = [(text, tooltip_text, _FONT_AWESOME_CLASSES[image_file], name_of_method) for (text, tooltip_text, image_file, name_of_method) in NavigationToolbar2.toolitems + (('Download', 'Download plot', 'download', 'download'),) if image_file in _FONT_AWESOME_CLASSES]

class FigureManagerNbAgg(FigureManagerWebAgg):
    _toolbar2_class = ToolbarCls = NavigationIPy

    def __init__(self, canvas, num):
        if False:
            for i in range(10):
                print('nop')
        self._shown = False
        super().__init__(canvas, num)

    @classmethod
    def create_with_canvas(cls, canvas_class, figure, num):
        if False:
            i = 10
            return i + 15
        canvas = canvas_class(figure)
        manager = cls(canvas, num)
        if is_interactive():
            manager.show()
            canvas.draw_idle()

        def destroy(event):
            if False:
                i = 10
                return i + 15
            canvas.mpl_disconnect(cid)
            Gcf.destroy(manager)
        cid = canvas.mpl_connect('close_event', destroy)
        return manager

    def display_js(self):
        if False:
            for i in range(10):
                print('nop')
        display(Javascript(FigureManagerNbAgg.get_javascript()))

    def show(self):
        if False:
            i = 10
            return i + 15
        if not self._shown:
            self.display_js()
            self._create_comm()
        else:
            self.canvas.draw_idle()
        self._shown = True
        if hasattr(self, '_cidgcf'):
            self.canvas.mpl_disconnect(self._cidgcf)
        if not is_interactive():
            from matplotlib._pylab_helpers import Gcf
            Gcf.figs.pop(self.num, None)

    def reshow(self):
        if False:
            i = 10
            return i + 15
        '\n        A special method to re-show the figure in the notebook.\n\n        '
        self._shown = False
        self.show()

    @property
    def connected(self):
        if False:
            for i in range(10):
                print('nop')
        return bool(self.web_sockets)

    @classmethod
    def get_javascript(cls, stream=None):
        if False:
            print('Hello World!')
        if stream is None:
            output = io.StringIO()
        else:
            output = stream
        super().get_javascript(stream=output)
        output.write((pathlib.Path(__file__).parent / 'web_backend/js/nbagg_mpl.js').read_text(encoding='utf-8'))
        if stream is None:
            return output.getvalue()

    def _create_comm(self):
        if False:
            i = 10
            return i + 15
        comm = CommSocket(self)
        self.add_web_socket(comm)
        return comm

    def destroy(self):
        if False:
            for i in range(10):
                print('nop')
        self._send_event('close')
        for comm in list(self.web_sockets):
            comm.on_close()
        self.clearup_closed()

    def clearup_closed(self):
        if False:
            while True:
                i = 10
        'Clear up any closed Comms.'
        self.web_sockets = {socket for socket in self.web_sockets if socket.is_open()}
        if len(self.web_sockets) == 0:
            CloseEvent('close_event', self.canvas)._process()

    def remove_comm(self, comm_id):
        if False:
            while True:
                i = 10
        self.web_sockets = {socket for socket in self.web_sockets if socket.comm.comm_id != comm_id}

class FigureCanvasNbAgg(FigureCanvasWebAggCore):
    manager_class = FigureManagerNbAgg

class CommSocket:
    """
    Manages the Comm connection between IPython and the browser (client).

    Comms are 2 way, with the CommSocket being able to publish a message
    via the send_json method, and handle a message with on_message. On the
    JS side figure.send_message and figure.ws.onmessage do the sending and
    receiving respectively.

    """

    def __init__(self, manager):
        if False:
            for i in range(10):
                print('nop')
        self.supports_binary = None
        self.manager = manager
        self.uuid = str(uuid.uuid4())
        display(HTML('<div id=%r></div>' % self.uuid))
        try:
            self.comm = Comm('matplotlib', data={'id': self.uuid})
        except AttributeError as err:
            raise RuntimeError('Unable to create an IPython notebook Comm instance. Are you in the IPython notebook?') from err
        self.comm.on_msg(self.on_message)
        manager = self.manager
        self._ext_close = False

        def _on_close(close_message):
            if False:
                for i in range(10):
                    print('nop')
            self._ext_close = True
            manager.remove_comm(close_message['content']['comm_id'])
            manager.clearup_closed()
        self.comm.on_close(_on_close)

    def is_open(self):
        if False:
            i = 10
            return i + 15
        return not (self._ext_close or self.comm._closed)

    def on_close(self):
        if False:
            return 10
        if self.is_open():
            try:
                self.comm.close()
            except KeyError:
                pass

    def send_json(self, content):
        if False:
            print('Hello World!')
        self.comm.send({'data': json.dumps(content)})

    def send_binary(self, blob):
        if False:
            print('Hello World!')
        if self.supports_binary:
            self.comm.send({'blob': 'image/png'}, buffers=[blob])
        else:
            data = b64encode(blob).decode('ascii')
            data_uri = f'data:image/png;base64,{data}'
            self.comm.send({'data': data_uri})

    def on_message(self, message):
        if False:
            for i in range(10):
                print('nop')
        message = json.loads(message['content']['data'])
        if message['type'] == 'closing':
            self.on_close()
            self.manager.clearup_closed()
        elif message['type'] == 'supports_binary':
            self.supports_binary = message['value']
        else:
            self.manager.handle_json(message)

@_Backend.export
class _BackendNbAgg(_Backend):
    FigureCanvas = FigureCanvasNbAgg
    FigureManager = FigureManagerNbAgg