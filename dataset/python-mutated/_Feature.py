"""Feature module.

Provides:
 - Feature - class to wrap Bio.SeqFeature objects with drawing information

For drawing capabilities, this module uses reportlab to define colors:
http://www.reportlab.com
"""
from reportlab.lib import colors
from ._Colors import ColorTranslator

class Feature:
    """Class to wrap Bio.SeqFeature objects for GenomeDiagram.

    Attributes:
     - parent    FeatureSet, container for the object
     - id        Unique id
     - color    color.Color, color to draw the feature
     - hide      Boolean for whether the feature will be drawn or not
     - sigil     String denoting the type of sigil to use for the feature.
       Currently either "BOX" or "ARROW" are supported.
     - arrowhead_length  Float denoting length of the arrow head to be drawn,
       relative to the bounding box height.  The arrow shaft
       takes up the remainder of the bounding box's length.
     - arrowshaft_height  Float denoting length of the representative arrow
       shaft to be drawn, relative to the bounding box height.
       The arrow head takes the full height of the bound box.
     - name_qualifiers   List of Strings, describes the qualifiers that may
       contain feature names in the wrapped Bio.SeqFeature object
     - label     Boolean, 1 if the label should be shown
     - label_font    String describing the font to use for the feature label
     - label_size    Int describing the feature label font size
     - label_color  color.Color describing the feature label color
     - label_angle   Float describing the angle through which to rotate the
       feature label in degrees (default = 45, linear only)
     - label_position    String, 'start', 'end' or 'middle' denoting where
       to place the feature label. Leave as None for the default
       which is 'start' for linear diagrams, and at the bottom of
       the feature as drawn on circular diagrams.
     - label_strand  Integer -1 or +1 to explicitly place the label on the
       forward or reverse strand. Default (None) follows th
       feature's strand. Use -1 to put labels under (linear) or
       inside (circular) the track, +1 to put them above (linear)
       or outside (circular) the track.
     - locations     List of tuples of (start, end) ints describing where the
       feature and any subfeatures start and end
     - type      String denoting the feature type
     - name      String denoting the feature name
     - strand    Int describing the strand on which the feature is found

    """

    def __init__(self, parent=None, feature_id=None, feature=None, color=colors.lightgreen, label=0, border=None, colour=None):
        if False:
            for i in range(10):
                print('nop')
        "Initialize.\n\n        Arguments:\n         - parent    FeatureSet containing the feature\n         - feature_id    Unique id for the feature\n         - feature   Bio.SeqFeature object to be wrapped\n         - color    color.Color Color to draw the feature (overridden\n           by backwards compatible argument with UK spelling, colour).\n           Either argument is overridden if 'color' is found in feature\n           qualifiers\n         - border   color.Color Color to draw the feature border, use\n           None for the same as the fill color, False for no border.\n         - label     Boolean, 1 if the label should be shown\n\n        "
        if colour is not None:
            color = colour
        self._colortranslator = ColorTranslator()
        self.parent = parent
        self.id = feature_id
        self.color = color
        self.border = border
        self._feature = None
        self.hide = 0
        self.sigil = 'BOX'
        self.arrowhead_length = 0.5
        self.arrowshaft_height = 0.4
        self.name_qualifiers = ['gene', 'label', 'name', 'locus_tag', 'product']
        self.label = label
        self.label_font = 'Helvetica'
        self.label_size = 6
        self.label_color = colors.black
        self.label_angle = 45
        self.label_position = None
        self.label_strand = None
        if feature is not None:
            self.set_feature(feature)

    def set_feature(self, feature):
        if False:
            for i in range(10):
                print('nop')
        'Define the Bio.SeqFeature object to be wrapped.'
        self._feature = feature
        self.__process_feature()

    def __process_feature(self):
        if False:
            while True:
                i = 10
        'Examine wrapped feature and set some properties accordingly (PRIVATE).'
        self.locations = []
        bounds = []
        for location in self._feature.location.parts:
            start = int(location.start)
            end = int(location.end)
            self.locations.append((start, end))
            bounds += [start, end]
        self.type = str(self._feature.type)
        if self._feature.strand is None:
            self.strand = 0
        else:
            self.strand = int(self._feature.strand)
        if 'color' in self._feature.qualifiers:
            self.color = self._colortranslator.artemis_color(self._feature.qualifiers['color'][0])
        self.name = self.type
        for qualifier in self.name_qualifiers:
            if qualifier in self._feature.qualifiers:
                self.name = self._feature.qualifiers[qualifier][0]
                break
        (self.start, self.end) = (min(bounds), max(bounds))

    def get_feature(self):
        if False:
            i = 10
            return i + 15
        'Return the unwrapped Bio.SeqFeature object.'
        return self._feature

    def set_colour(self, colour):
        if False:
            i = 10
            return i + 15
        'Backwards compatible variant of set_color(self, color) using UK spelling.'
        color = self._colortranslator.translate(colour)
        self.color = color

    def set_color(self, color):
        if False:
            while True:
                i = 10
        'Set the color in which the feature will be drawn.\n\n        Arguments:\n         - color    The color to draw the feature - either a colors.Color\n           object, an RGB tuple of floats, or an integer corresponding a\n           colors in colors.txt\n\n        '
        color = self._colortranslator.translate(color)
        self.color = color

    def __getattr__(self, name):
        if False:
            for i in range(10):
                print('nop')
        "Get attribute by name.\n\n        If the Feature class doesn't have the attribute called for,\n        check in self._feature for it.\n        "
        return getattr(self._feature, name)
if __name__ == '__main__':
    gdf = Feature()