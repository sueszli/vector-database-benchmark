"""A logging handler class which produces coloured logs."""
import logging
import click

class ColorHandler(logging.StreamHandler):
    """A color log handler.

    You can use this handler by incorporating the example below into your
    logging configuration:

    ``conf/project/logging.yml``:
    ::

        formatters:
          simple:
            format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

        handlers:
          console:
            class: kedro.extras.logging.ColorHandler
            level: INFO
            formatter: simple
            stream: ext://sys.stdout
            # defining colors is optional
            colors:
              debug: white
              info: magenta
              warning: yellow

        root:
          level: INFO
          handlers: [console]

    The ``colors`` parameter is optional, and you can use any ANSI color.

      * Black
      * Red
      * Green
      * Yellow
      * Blue
      * Magenta
      * Cyan
      * White

    The default colors are:

      * debug: magenta
      * info: cyan
      * warning: yellow
      * error: red
      * critical: red
    """

    def __init__(self, stream=None, colors=None):
        if False:
            print('Hello World!')
        logging.StreamHandler.__init__(self, stream)
        colors = colors or {}
        self.colors = {'critical': colors.get('critical', 'red'), 'error': colors.get('error', 'red'), 'warning': colors.get('warning', 'yellow'), 'info': colors.get('info', 'cyan'), 'debug': colors.get('debug', 'magenta')}

    def _get_color(self, level):
        if False:
            return 10
        if level >= logging.CRITICAL:
            return self.colors['critical']
        if level >= logging.ERROR:
            return self.colors['error']
        if level >= logging.WARNING:
            return self.colors['warning']
        if level >= logging.INFO:
            return self.colors['info']
        if level >= logging.DEBUG:
            return self.colors['debug']
        return None

    def format(self, record: logging.LogRecord) -> str:
        if False:
            return 10
        'The handler formatter.\n\n        Args:\n            record: The record to format.\n\n        Returns:\n            The record formatted as a string.\n\n        '
        text = logging.StreamHandler.format(self, record)
        color = self._get_color(record.levelno)
        return click.style(text, color)