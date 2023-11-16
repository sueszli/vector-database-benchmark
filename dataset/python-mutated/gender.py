"""gender(first_name): return predicted gender.  Requires gender-guesser library.

Possible values:
- "unknown" (name not found)
- "andy" (androgynous)
- "male"
- "female"
- "mostly_male"
- "mostly_female"
"""
import functools

@functools.lru_cache()
def gg_detect():
    if False:
        return 10
    import gender_guesser.detector
    return gender_guesser.detector.Detector()

def gender(first_name):
    if False:
        i = 10
        return i + 15
    return gg_detect().get_gender(first_name.capitalize())