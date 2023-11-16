from moviepy.decorators import apply_to_audio, apply_to_mask, requires_duration

@requires_duration
@apply_to_mask
@apply_to_audio
def time_mirror(clip):
    if False:
        i = 10
        return i + 15
    "\n    Returns a clip that plays the current clip backwards.\n    The clip must have its ``duration`` attribute set.\n    The same effect is applied to the clip's audio and mask if any.\n    "
    return clip[::-1]