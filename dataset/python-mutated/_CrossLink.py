"""Genome Diagram Feature cross-link module."""
from reportlab.lib import colors

class CrossLink:
    """Hold information for drawing a cross link between features."""

    def __init__(self, featureA, featureB, color=colors.lightgreen, border=None, flip=False):
        if False:
            return 10
        'Create a new cross link.\n\n        Arguments featureA and featureB should GenomeDiagram feature objects,\n        or 3-tuples (track object, start, end), and currently must be on\n        different tracks.\n\n        The color and border arguments should be ReportLab colour objects, or\n        for border use a boolean False for no border, otherwise it defaults to\n        the same as the main colour.\n\n        The flip argument draws an inverted cross link, useful for showing a\n        mapping where one sequence has been reversed. It is conventional to\n        also use a different colour (e.g. red for simple links, blue for any\n        flipped links).\n        '
        self.featureA = featureA
        self.featureB = featureB
        self.color = color
        self.border = border
        self.flip = flip

    @property
    def startA(self):
        if False:
            for i in range(10):
                print('nop')
        'Start position of Feature A.'
        try:
            return self.featureA.start
        except AttributeError:
            (track, start, end) = self.featureA
            return start

    @property
    def endA(self):
        if False:
            for i in range(10):
                print('nop')
        'End position of Feature A.'
        try:
            return self.featureA.end
        except AttributeError:
            (track, start, end) = self.featureA
            return end

    def _trackA(self, tracks):
        if False:
            for i in range(10):
                print('nop')
        try:
            (track, start, end) = self.featureA
            assert track in tracks
            return track
        except TypeError:
            for track in tracks:
                for feature_set in track.get_sets():
                    if hasattr(feature_set, 'features'):
                        if self.featureA in feature_set.features.values():
                            return track
            return None

    @property
    def startB(self):
        if False:
            for i in range(10):
                print('nop')
        'Start position of Feature B.'
        try:
            return self.featureB.start
        except AttributeError:
            (track, start, end) = self.featureB
            return start

    @property
    def endB(self):
        if False:
            i = 10
            return i + 15
        'End position of Feature B.'
        try:
            return self.featureB.end
        except AttributeError:
            (track, start, end) = self.featureB
            return end

    def _trackB(self, tracks):
        if False:
            return 10
        try:
            (track, start, end) = self.featureB
            assert track in tracks
            return track
        except TypeError:
            for track in tracks:
                for feature_set in track.get_sets():
                    if hasattr(feature_set, 'features'):
                        if self.featureB in feature_set.features.values():
                            return track
            return None