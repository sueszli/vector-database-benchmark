"""An interface for publishing rich data to frontends.

There are two components of the display system:

* Display formatters, which take a Python object and compute the
  representation of the object in various formats (text, HTML, SVG, etc.).
* The display publisher that is used to send the representation data to the
  various frontends.

This module defines the logic display publishing. The display publisher uses
the ``display_data`` message type that is defined in the IPython messaging
spec.
"""
import sys
from traitlets.config.configurable import Configurable
from traitlets import List
from .display_functions import publish_display_data

class DisplayPublisher(Configurable):
    """A traited class that publishes display data to frontends.

    Instances of this class are created by the main IPython object and should
    be accessed there.
    """

    def __init__(self, shell=None, *args, **kwargs):
        if False:
            return 10
        self.shell = shell
        super().__init__(*args, **kwargs)

    def _validate_data(self, data, metadata=None):
        if False:
            i = 10
            return i + 15
        'Validate the display data.\n\n        Parameters\n        ----------\n        data : dict\n            The formata data dictionary.\n        metadata : dict\n            Any metadata for the data.\n        '
        if not isinstance(data, dict):
            raise TypeError('data must be a dict, got: %r' % data)
        if metadata is not None:
            if not isinstance(metadata, dict):
                raise TypeError('metadata must be a dict, got: %r' % data)

    def publish(self, data, metadata=None, source=None, *, transient=None, update=False, **kwargs) -> None:
        if False:
            return 10
        "Publish data and metadata to all frontends.\n\n        See the ``display_data`` message in the messaging documentation for\n        more details about this message type.\n\n        The following MIME types are currently implemented:\n\n        * text/plain\n        * text/html\n        * text/markdown\n        * text/latex\n        * application/json\n        * application/javascript\n        * image/png\n        * image/jpeg\n        * image/svg+xml\n\n        Parameters\n        ----------\n        data : dict\n            A dictionary having keys that are valid MIME types (like\n            'text/plain' or 'image/svg+xml') and values that are the data for\n            that MIME type. The data itself must be a JSON'able data\n            structure. Minimally all data should have the 'text/plain' data,\n            which can be displayed by all frontends. If more than the plain\n            text is given, it is up to the frontend to decide which\n            representation to use.\n        metadata : dict\n            A dictionary for metadata related to the data. This can contain\n            arbitrary key, value pairs that frontends can use to interpret\n            the data.  Metadata specific to each mime-type can be specified\n            in the metadata dict with the same mime-type keys as\n            the data itself.\n        source : str, deprecated\n            Unused.\n        transient : dict, keyword-only\n            A dictionary for transient data.\n            Data in this dictionary should not be persisted as part of saving this output.\n            Examples include 'display_id'.\n        update : bool, keyword-only, default: False\n            If True, only update existing outputs with the same display_id,\n            rather than creating a new output.\n        "
        handlers = {}
        if self.shell is not None:
            handlers = getattr(self.shell, 'mime_renderers', {})
        for (mime, handler) in handlers.items():
            if mime in data:
                handler(data[mime], metadata.get(mime, None))
                return
        if 'text/plain' in data:
            print(data['text/plain'])

    def clear_output(self, wait=False):
        if False:
            while True:
                i = 10
        'Clear the output of the cell receiving output.'
        print('\x1b[2K\r', end='')
        sys.stdout.flush()
        print('\x1b[2K\r', end='')
        sys.stderr.flush()

class CapturingDisplayPublisher(DisplayPublisher):
    """A DisplayPublisher that stores"""
    outputs = List()

    def publish(self, data, metadata=None, source=None, *, transient=None, update=False):
        if False:
            for i in range(10):
                print('nop')
        self.outputs.append({'data': data, 'metadata': metadata, 'transient': transient, 'update': update})

    def clear_output(self, wait=False):
        if False:
            i = 10
            return i + 15
        super(CapturingDisplayPublisher, self).clear_output(wait)
        self.outputs.clear()