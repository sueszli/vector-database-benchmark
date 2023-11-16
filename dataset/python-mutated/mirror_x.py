def mirror_x(clip, apply_to='mask'):
    if False:
        while True:
            i = 10
    'Flips the clip horizontally (and its mask too, by default).'
    return clip.image_transform(lambda img: img[:, ::-1], apply_to=apply_to)