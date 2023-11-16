from mutagen.id3 import ID3, Frames, Frames_2_2, TextFrame
try:
    from mutagen.id3 import GRP1
except ImportError:

    class GRP1(TextFrame):
        pass

class XSOP(TextFrame):
    pass
known_frames = dict(Frames)
known_frames.update(dict(Frames_2_2))
known_frames['GRP1'] = GRP1
known_frames['XSOP'] = XSOP

class CompatID3(ID3):
    """
    Additional features over mutagen.id3.ID3:
     * Allow some v2.4 frames also in v2.3
    """
    PEDANTIC = False

    def __init__(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        if args:
            kwargs['known_frames'] = known_frames
        super().__init__(*args, **kwargs)

    def update_to_v23(self):
        if False:
            i = 10
            return i + 15
        update_to_v23(self)

def update_to_v23(tags):
    if False:
        return 10
    frames = []
    for key in {'TSOP', 'TSOA', 'TSOT', 'TSST'}:
        frames.extend(tags.getall(key))
    ID3.update_to_v23(tags)
    for frame in frames:
        tags.add(frame)