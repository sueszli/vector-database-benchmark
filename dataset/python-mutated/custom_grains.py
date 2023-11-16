def test(grains):
    if False:
        i = 10
        return i + 15
    return {'custom_grain_test': 'itworked' if 'os' in grains else 'itdidntwork'}