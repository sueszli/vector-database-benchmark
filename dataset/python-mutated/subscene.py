from __future__ import division
from .node import Node

class SubScene(Node):
    """A Node subclass that serves as a marker and parent node for certain
    branches of the scenegraph.

    SubScene nodes are used as the top-level node for the internal scenes of
    a canvas and a view box.
    """

    def __init__(self, **kwargs):
        if False:
            while True:
                i = 10
        Node.__init__(self, **kwargs)
        self.document = self