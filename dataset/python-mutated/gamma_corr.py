def gamma_corr(clip, gamma):
    if False:
        for i in range(10):
            print('nop')
    'Gamma-correction of a video clip.'

    def filter(im):
        if False:
            return 10
        corrected = 255 * (1.0 * im / 255) ** gamma
        return corrected.astype('uint8')
    return clip.image_transform(filter)