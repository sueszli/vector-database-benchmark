"""
================
Embedding WebAgg
================

This example demonstrates how to embed Matplotlib WebAgg interactive plotting
in your own web application and framework.  It is not necessary to do all this
if you merely want to display a plot in a browser or use Matplotlib's built-in
Tornado-based server "on the side".

The framework being used must support web sockets.
"""
import argparse
import io
import json
import mimetypes
from pathlib import Path
import signal
import socket
try:
    import tornado
except ImportError as err:
    raise RuntimeError('This example requires tornado.') from err
import tornado.httpserver
import tornado.ioloop
import tornado.web
import tornado.websocket
import numpy as np
import matplotlib as mpl
from matplotlib.backends.backend_webagg import FigureManagerWebAgg, new_figure_manager_given_figure
from matplotlib.figure import Figure

def create_figure():
    if False:
        print('Hello World!')
    '\n    Creates a simple example figure.\n    '
    fig = Figure()
    ax = fig.add_subplot()
    t = np.arange(0.0, 3.0, 0.01)
    s = np.sin(2 * np.pi * t)
    ax.plot(t, s)
    return fig
html_content = '<!DOCTYPE html>\n<html lang="en">\n  <head>\n    <!-- TODO: There should be a way to include all of the required javascript\n               and CSS so matplotlib can add to the set in the future if it\n               needs to. -->\n    <link rel="stylesheet" href="_static/css/page.css" type="text/css">\n    <link rel="stylesheet" href="_static/css/boilerplate.css" type="text/css">\n    <link rel="stylesheet" href="_static/css/fbm.css" type="text/css">\n    <link rel="stylesheet" href="_static/css/mpl.css" type="text/css">\n    <script src="mpl.js"></script>\n\n    <script>\n      /* This is a callback that is called when the user saves\n         (downloads) a file.  Its purpose is really to map from a\n         figure and file format to a url in the application. */\n      function ondownload(figure, format) {\n        window.open(\'download.\' + format, \'_blank\');\n      };\n\n      function ready(fn) {\n        if (document.readyState != "loading") {\n          fn();\n        } else {\n          document.addEventListener("DOMContentLoaded", fn);\n        }\n      }\n\n      ready(\n        function() {\n          /* It is up to the application to provide a websocket that the figure\n             will use to communicate to the server.  This websocket object can\n             also be a "fake" websocket that underneath multiplexes messages\n             from multiple figures, if necessary. */\n          var websocket_type = mpl.get_websocket_type();\n          var websocket = new websocket_type("%(ws_uri)sws");\n\n          // mpl.figure creates a new figure on the webpage.\n          var fig = new mpl.figure(\n              // A unique numeric identifier for the figure\n              %(fig_id)s,\n              // A websocket object (or something that behaves like one)\n              websocket,\n              // A function called when a file type is selected for download\n              ondownload,\n              // The HTML element in which to place the figure\n              document.getElementById("figure"));\n        }\n      );\n    </script>\n\n    <title>matplotlib</title>\n  </head>\n\n  <body>\n    <div id="figure">\n    </div>\n  </body>\n</html>\n'

class MyApplication(tornado.web.Application):

    class MainPage(tornado.web.RequestHandler):
        """
        Serves the main HTML page.
        """

        def get(self):
            if False:
                print('Hello World!')
            manager = self.application.manager
            ws_uri = f'ws://{self.request.host}/'
            content = html_content % {'ws_uri': ws_uri, 'fig_id': manager.num}
            self.write(content)

    class MplJs(tornado.web.RequestHandler):
        """
        Serves the generated matplotlib javascript file.  The content
        is dynamically generated based on which toolbar functions the
        user has defined.  Call `FigureManagerWebAgg` to get its
        content.
        """

        def get(self):
            if False:
                return 10
            self.set_header('Content-Type', 'application/javascript')
            js_content = FigureManagerWebAgg.get_javascript()
            self.write(js_content)

    class Download(tornado.web.RequestHandler):
        """
        Handles downloading of the figure in various file formats.
        """

        def get(self, fmt):
            if False:
                return 10
            manager = self.application.manager
            self.set_header('Content-Type', mimetypes.types_map.get(fmt, 'binary'))
            buff = io.BytesIO()
            manager.canvas.figure.savefig(buff, format=fmt)
            self.write(buff.getvalue())

    class WebSocket(tornado.websocket.WebSocketHandler):
        """
        A websocket for interactive communication between the plot in
        the browser and the server.

        In addition to the methods required by tornado, it is required to
        have two callback methods:

            - ``send_json(json_content)`` is called by matplotlib when
              it needs to send json to the browser.  `json_content` is
              a JSON tree (Python dictionary), and it is the responsibility
              of this implementation to encode it as a string to send over
              the socket.

            - ``send_binary(blob)`` is called to send binary image data
              to the browser.
        """
        supports_binary = True

        def open(self):
            if False:
                for i in range(10):
                    print('nop')
            manager = self.application.manager
            manager.add_web_socket(self)
            if hasattr(self, 'set_nodelay'):
                self.set_nodelay(True)

        def on_close(self):
            if False:
                for i in range(10):
                    print('nop')
            manager = self.application.manager
            manager.remove_web_socket(self)

        def on_message(self, message):
            if False:
                while True:
                    i = 10
            message = json.loads(message)
            if message['type'] == 'supports_binary':
                self.supports_binary = message['value']
            else:
                manager = self.application.manager
                manager.handle_json(message)

        def send_json(self, content):
            if False:
                return 10
            self.write_message(json.dumps(content))

        def send_binary(self, blob):
            if False:
                print('Hello World!')
            if self.supports_binary:
                self.write_message(blob, binary=True)
            else:
                data_uri = 'data:image/png;base64,' + blob.encode('base64').replace('\n', '')
                self.write_message(data_uri)

    def __init__(self, figure):
        if False:
            for i in range(10):
                print('nop')
        self.figure = figure
        self.manager = new_figure_manager_given_figure(id(figure), figure)
        super().__init__([('/_static/(.*)', tornado.web.StaticFileHandler, {'path': FigureManagerWebAgg.get_static_file_path()}), ('/_images/(.*)', tornado.web.StaticFileHandler, {'path': Path(mpl.get_data_path(), 'images')}), ('/', self.MainPage), ('/mpl.js', self.MplJs), ('/ws', self.WebSocket), ('/download.([a-z0-9.]+)', self.Download)])
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--port', type=int, default=8080, help='Port to listen on (0 for a random port).')
    args = parser.parse_args()
    figure = create_figure()
    application = MyApplication(figure)
    http_server = tornado.httpserver.HTTPServer(application)
    sockets = tornado.netutil.bind_sockets(args.port, '')
    http_server.add_sockets(sockets)
    for s in sockets:
        (addr, port) = s.getsockname()[:2]
        if s.family is socket.AF_INET6:
            addr = f'[{addr}]'
        print(f'Listening on http://{addr}:{port}/')
    print('Press Ctrl+C to quit')
    ioloop = tornado.ioloop.IOLoop.instance()

    def shutdown():
        if False:
            while True:
                i = 10
        ioloop.stop()
        print('Server stopped')
    old_handler = signal.signal(signal.SIGINT, lambda sig, frame: ioloop.add_callback_from_signal(shutdown))
    try:
        ioloop.start()
    finally:
        signal.signal(signal.SIGINT, old_handler)