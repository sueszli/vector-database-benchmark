def invert_colors(clip):
    if False:
        while True:
            i = 10
    'Returns the color-inversed clip.\n\n    The values of all pixels are replaced with (255-v) or (1-v) for masks\n    Black becomes white, green becomes purple, etc.\n    '
    maxi = 1.0 if clip.is_mask else 255
    return clip.image_transform(lambda f: maxi - f)