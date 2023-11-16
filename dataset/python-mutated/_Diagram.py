"""Provides a container for information concerning the tracks to be drawn in a diagram.

It also provides the interface for defining the diagram (possibly split these
functions in later version?).

For drawing capabilities, this module uses reportlab to draw and write the
diagram:

http://www.reportlab.com

For dealing with biological information, the package expects BioPython
objects - namely SeqRecord objects containing SeqFeature objects.
"""
try:
    from reportlab.graphics import renderPM
except ImportError:
    renderPM = None
from ._LinearDrawer import LinearDrawer
from ._CircularDrawer import CircularDrawer
from ._Track import Track
from Bio.Graphics import _write

def _first_defined(*args):
    if False:
        print('Hello World!')
    'Return the first non-null argument (PRIVATE).'
    for arg in args:
        if arg is not None:
            return arg
    return None

class Diagram:
    """Diagram container.

    Arguments:
     - name           - a string, identifier for the diagram.
     - tracks         - a list of Track objects comprising the diagram.
     - format         - a string, format of the diagram 'circular' or
       'linear', depending on the sort of diagram required.
     - pagesize       - a string, the pagesize of output describing the ISO
       size of the image, or a tuple of pixels.
     - orientation    - a string describing the required orientation of the
       final drawing ('landscape' or 'portrait').
     - x              - a float (0->1), the proportion of the page to take
       up with even X margins t the page.
     - y              - a float (0->1), the proportion of the page to take
       up with even Y margins to the page.
     - xl             - a float (0->1), the proportion of the page to take
       up with the left X margin to the page (overrides x).
     - xr             - a float (0->1), the proportion of the page to take
       up with the right X margin to the page (overrides x).
     - yt             - a float (0->1), the proportion of the page to take
       up with the top Y margin to the page (overrides y).
     - yb             - a float (0->1), the proportion of the page to take
       up with the bottom Y margin to the page (overrides y).
     - circle_core    - a float, the proportion of the available radius to
       leave empty at the center of a circular diagram (0 to 1).
     - start          - an integer, the base/aa position to start the diagram at.
     - end            - an integer, the base/aa position to end the diagram at.
     - tracklines     - a boolean, True if track guidelines are to be drawn.
     - fragments      - and integer, for a linear diagram, the number of equal
       divisions into which the sequence is divided.
     - fragment_size  - a float (0->1), the proportion of the space
       available to each fragment that should be used in drawing.
     - track_size     - a float (0->1), the proportion of the space
       available to each track that should be used in drawing with sigils.
     - circular       - a boolean, True if the genome/sequence to be drawn
       is, in reality, circular.

    """

    def __init__(self, name=None, format='circular', pagesize='A3', orientation='landscape', x=0.05, y=0.05, xl=None, xr=None, yt=None, yb=None, start=None, end=None, tracklines=False, fragments=10, fragment_size=None, track_size=0.75, circular=True, circle_core=0.0):
        if False:
            while True:
                i = 10
        'Initialize.\n\n        gdd = Diagram(name=None)\n        '
        self.tracks = {}
        self.name = name
        self.format = format
        self.pagesize = pagesize
        self.orientation = orientation
        self.x = x
        self.y = y
        self.xl = xl
        self.xr = xr
        self.yt = yt
        self.yb = yb
        self.start = start
        self.end = end
        self.tracklines = tracklines
        self.fragments = fragments
        if fragment_size is not None:
            self.fragment_size = fragment_size
        elif self.fragments == 1:
            self.fragment_size = 1
        else:
            self.fragment_size = 0.9
        self.track_size = track_size
        self.circular = circular
        self.circle_core = circle_core
        self.cross_track_links = []
        self.drawing = None

    def set_all_tracks(self, attr, value):
        if False:
            i = 10
            return i + 15
        'Set the passed attribute of all tracks in the set to the passed value.\n\n        Arguments:\n         - attr    - An attribute of the Track class.\n         - value   - The value to set that attribute.\n\n        set_all_tracks(self, attr, value)\n        '
        for track in self.tracks.values():
            if hasattr(track, attr):
                setattr(track, attr, value)

    def draw(self, format=None, pagesize=None, orientation=None, x=None, y=None, xl=None, xr=None, yt=None, yb=None, start=None, end=None, tracklines=None, fragments=None, fragment_size=None, track_size=None, circular=None, circle_core=None, cross_track_links=None):
        if False:
            return 10
        "Draw the diagram, with passed parameters overriding existing attributes.\n\n        gdd.draw(format='circular')\n        "
        if format == 'linear':
            drawer = LinearDrawer(self, _first_defined(pagesize, self.pagesize), _first_defined(orientation, self.orientation), _first_defined(x, self.x), _first_defined(y, self.y), _first_defined(xl, self.xl), _first_defined(xr, self.xr), _first_defined(yt, self.yt), _first_defined(yb, self.yb), _first_defined(start, self.start), _first_defined(end, self.end), _first_defined(tracklines, self.tracklines), _first_defined(fragments, self.fragments), _first_defined(fragment_size, self.fragment_size), _first_defined(track_size, self.track_size), _first_defined(cross_track_links, self.cross_track_links))
        else:
            drawer = CircularDrawer(self, _first_defined(pagesize, self.pagesize), _first_defined(orientation, self.orientation), _first_defined(x, self.x), _first_defined(y, self.y), _first_defined(xl, self.xl), _first_defined(xr, self.xr), _first_defined(yt, self.yt), _first_defined(yb, self.yb), _first_defined(start, self.start), _first_defined(end, self.end), _first_defined(tracklines, self.tracklines), _first_defined(track_size, self.track_size), _first_defined(circular, self.circular), _first_defined(circle_core, self.circle_core), _first_defined(cross_track_links, self.cross_track_links))
        drawer.draw()
        self.drawing = drawer.drawing

    def write(self, filename='test1.ps', output='PS', dpi=72):
        if False:
            while True:
                i = 10
        "Write the drawn diagram to a specified file, in a specified format.\n\n        Arguments:\n            - filename   - a string indicating the name of the output file,\n              or a handle to write to.\n            - output     - a string indicating output format, one of PS, PDF,\n              SVG, or provided the ReportLab renderPM module is installed, one\n              of the bitmap formats JPG, BMP, GIF, PNG, TIFF or TIFF.  The\n              format can be given in upper or lower case.\n            - dpi        - an integer. Resolution (dots per inch) for bitmap formats.\n\n        Returns:\n            No return value.\n\n        write(self, filename='test1.ps', output='PS', dpi=72)\n\n        "
        return _write(self.drawing, filename, output, dpi=dpi)

    def write_to_string(self, output='PS', dpi=72):
        if False:
            for i in range(10):
                print('nop')
        'Return a byte string containing the diagram in the requested format.\n\n        Arguments:\n            - output    - a string indicating output format, one of PS, PDF,\n              SVG, JPG, BMP, GIF, PNG, TIFF or TIFF (as specified for the write\n              method).\n            - dpi       - Resolution (dots per inch) for bitmap formats.\n\n        Returns:\n            Return the completed drawing as a bytes string in a prescribed\n            format.\n\n        '
        from io import BytesIO
        handle = BytesIO()
        self.write(handle, output, dpi)
        return handle.getvalue()

    def add_track(self, track, track_level):
        if False:
            while True:
                i = 10
        'Add a Track object to the diagram.\n\n        It also accepts instructions to place it at a particular level on the\n        diagram.\n\n        Arguments:\n            - track          - Track object to draw.\n            - track_level    - an integer. The level at which the track will be\n              drawn (above an arbitrary baseline).\n\n        add_track(self, track, track_level)\n        '
        if track is None:
            raise ValueError('Must specify track')
        if track_level not in self.tracks:
            self.tracks[track_level] = track
        else:
            occupied_levels = sorted(self.get_levels())
            occupied_levels.reverse()
            for val in occupied_levels:
                if val >= track.track_level:
                    self.tracks[val + 1] = self.tracks[val]
            self.tracks[track_level] = track
        self.tracks[track_level].track_level = track_level

    def new_track(self, track_level, **args):
        if False:
            print('Hello World!')
        'Add a new Track to the diagram at a given level.\n\n        The track is returned for further user manipulation.\n\n        Arguments:\n            - track_level   - an integer. The level at which the track will be\n              drawn (above an arbitrary baseline).\n\n        new_track(self, track_level)\n        '
        newtrack = Track()
        for key in args:
            setattr(newtrack, key, args[key])
        if track_level not in self.tracks:
            self.tracks[track_level] = newtrack
        else:
            occupied_levels = sorted(self.get_levels())
            occupied_levels.reverse()
            for val in occupied_levels:
                if val >= track_level:
                    self.tracks[val + 1] = self.tracks[val]
            self.tracks[track_level] = newtrack
        self.tracks[track_level].track_level = track_level
        return newtrack

    def del_track(self, track_level):
        if False:
            print('Hello World!')
        'Remove the track to be drawn at a particular level on the diagram.\n\n        Arguments:\n            - track_level   - an integer. The level of the track on the diagram\n              to delete.\n\n        del_track(self, track_level)\n        '
        del self.tracks[track_level]

    def get_tracks(self):
        if False:
            return 10
        'Return a list of the tracks contained in the diagram.'
        return list(self.tracks.values())

    def move_track(self, from_level, to_level):
        if False:
            i = 10
            return i + 15
        'Move a track from one level on the diagram to another.\n\n        Arguments:\n         - from_level   - an integer. The level at which the track to be\n           moved is found.\n         - to_level     - an integer. The level to move the track to.\n\n        '
        aux = self.tracks[from_level]
        del self.tracks[from_level]
        self.add_track(aux, to_level)

    def renumber_tracks(self, low=1, step=1):
        if False:
            for i in range(10):
                print('nop')
        'Renumber all tracks consecutively.\n\n        Optionally from a passed lowest number.\n\n        Arguments:\n         - low     - an integer. The track number to start from.\n         - step    - an integer. The track interval for separation of\n           tracks.\n\n        '
        track = low
        levels = self.get_levels()
        conversion = {}
        for level in levels:
            conversion[track] = self.tracks[level]
            conversion[track].track_level = track
            track += step
        self.tracks = conversion

    def get_levels(self):
        if False:
            while True:
                i = 10
        'Return a sorted list of levels occupied by tracks in the diagram.'
        return sorted(self.tracks)

    def get_drawn_levels(self):
        if False:
            i = 10
            return i + 15
        'Return a sorted list of levels occupied by tracks.\n\n        These tracks are not explicitly hidden.\n        '
        return sorted((key for key in self.tracks if not self.tracks[key].hide))

    def range(self):
        if False:
            print('Hello World!')
        'Return lowest and highest base numbers from track features.\n\n        Returned type is a tuple.\n        '
        (lows, highs) = ([], [])
        for track in self.tracks.values():
            (low, high) = track.range()
            lows.append(low)
            highs.append(high)
        return (min(lows), max(highs))

    def __getitem__(self, key):
        if False:
            while True:
                i = 10
        'Return the track contained at the level of the passed key.'
        return self.tracks[key]

    def __str__(self):
        if False:
            for i in range(10):
                print('nop')
        'Return a formatted string describing the diagram.'
        outstr = [f'\n<{self.__class__}: {self.name}>']
        outstr.append('%d tracks' % len(self.tracks))
        for level in self.get_levels():
            outstr.append('Track %d: %s\n' % (level, self.tracks[level]))
        outstr = '\n'.join(outstr)
        return outstr