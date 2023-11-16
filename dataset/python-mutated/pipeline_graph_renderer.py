"""For rendering pipeline graph in HTML-compatible format.

This module is experimental. No backwards-compatibility guarantees.
"""
import abc
import os
import subprocess
from typing import TYPE_CHECKING
from typing import Optional
from typing import Type
from apache_beam.utils.plugin import BeamPlugin
if TYPE_CHECKING:
    from apache_beam.runners.interactive.display.pipeline_graph import PipelineGraph

class PipelineGraphRenderer(BeamPlugin, metaclass=abc.ABCMeta):
    """Abstract class for renderers, who decide how pipeline graphs are rendered.
  """

    @classmethod
    @abc.abstractmethod
    def option(cls):
        if False:
            i = 10
            return i + 15
        'The corresponding rendering option for the renderer.\n    '
        raise NotImplementedError

    @abc.abstractmethod
    def render_pipeline_graph(self, pipeline_graph):
        if False:
            for i in range(10):
                print('nop')
        'Renders the pipeline graph in HTML-compatible format.\n\n    Args:\n      pipeline_graph: (pipeline_graph.PipelineGraph) the graph to be rendererd.\n\n    Returns:\n      unicode, str or bytes that can be expressed as HTML.\n    '
        raise NotImplementedError

class MuteRenderer(PipelineGraphRenderer):
    """Use this renderer to mute the pipeline display.
  """

    @classmethod
    def option(cls):
        if False:
            while True:
                i = 10
        return 'mute'

    def render_pipeline_graph(self, pipeline_graph):
        if False:
            print('Hello World!')
        return ''

class TextRenderer(PipelineGraphRenderer):
    """This renderer simply returns the dot representation in text format.
  """

    @classmethod
    def option(cls):
        if False:
            for i in range(10):
                print('nop')
        return 'text'

    def render_pipeline_graph(self, pipeline_graph):
        if False:
            i = 10
            return i + 15
        return pipeline_graph.get_dot()

class PydotRenderer(PipelineGraphRenderer):
    """This renderer renders the graph using pydot.

  It depends on
    1. The software Graphviz: https://www.graphviz.org/
    2. The python module pydot: https://pypi.org/project/pydot/
  """

    @classmethod
    def option(cls):
        if False:
            return 10
        return 'graph'

    def render_pipeline_graph(self, pipeline_graph):
        if False:
            return 10
        return pipeline_graph._get_graph().create_svg().decode('utf-8')

def get_renderer(option=None):
    if False:
        i = 10
        return i + 15
    'Get an instance of PipelineGraphRenderer given rendering option.\n\n  Args:\n    option: (str) the rendering option.\n\n  Returns:\n    (PipelineGraphRenderer)\n  '
    if option is None:
        if os.name == 'nt':
            exists = subprocess.call(['where', 'dot.exe']) == 0
        else:
            exists = subprocess.call(['which', 'dot']) == 0
        if exists:
            option = 'graph'
        else:
            option = 'text'
    renderer = [r for r in PipelineGraphRenderer.get_all_subclasses() if option == r.option()]
    if len(renderer) == 0:
        raise ValueError()
    elif len(renderer) == 1:
        return renderer[0]()
    else:
        raise ValueError('Found more than one renderer for option: %s', option)