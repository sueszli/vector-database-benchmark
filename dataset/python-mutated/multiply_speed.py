def multiply_speed(clip, factor=None, final_duration=None):
    if False:
        while True:
            i = 10
    "Returns a clip playing the current clip but at a speed multiplied by ``factor``.\n\n    Instead of factor one can indicate the desired ``final_duration`` of the clip, and\n    the factor will be automatically computed. The same effect is applied to the clip's\n    audio and mask if any.\n    "
    if final_duration:
        factor = 1.0 * clip.duration / final_duration
    new_clip = clip.time_transform(lambda t: factor * t, apply_to=['mask', 'audio'])
    if clip.duration is not None:
        new_clip = new_clip.with_duration(1.0 * clip.duration / factor)
    return new_clip