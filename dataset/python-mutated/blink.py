def blink(clip, duration_on, duration_off):
    if False:
        return 10
    '\n    Makes the clip blink. At each blink it will be displayed ``duration_on``\n    seconds and disappear ``duration_off`` seconds. Will only work in\n    composite clips.\n    '
    new_clip = clip.copy()
    if new_clip.mask is None:
        new_clip = new_clip.with_mask()
    duration = duration_on + duration_off
    new_clip.mask = new_clip.mask.transform(lambda get_frame, t: get_frame(t) * (t % duration < duration_on))
    return new_clip