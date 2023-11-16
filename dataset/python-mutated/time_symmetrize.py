from moviepy.decorators import requires_duration

@requires_duration
def time_symmetrize(clip):
    if False:
        return 10
    "\n    Returns a clip that plays the current clip once forwards and\n    then once backwards. This is very practival to make video that\n    loop well, e.g. to create animated GIFs.\n    This effect is automatically applied to the clip's mask and audio\n    if they exist.\n    "
    return clip + clip[::-1]