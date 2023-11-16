from msrest.serialization import Model

class DetectedLanguage(Model):
    """DetectedLanguage.

    :param name: Long name of a detected language (e.g. English, French).
    :type name: str
    :param iso6391_name: A two letter representation of the detected language
     according to the ISO 639-1 standard (e.g. en, fr).
    :type iso6391_name: str
    :param score: A confidence score between 0 and 1. Scores close to 1
     indicate 100% certainty that the identified language is true.
    :type score: float
    """
    _attribute_map = {'name': {'key': 'name', 'type': 'str'}, 'iso6391_name': {'key': 'iso6391Name', 'type': 'str'}, 'score': {'key': 'score', 'type': 'float'}}

    def __init__(self, *, name: str=None, iso6391_name: str=None, score: float=None, **kwargs) -> None:
        if False:
            while True:
                i = 10
        super(DetectedLanguage, self).__init__(**kwargs)
        self.name = name
        self.iso6391_name = iso6391_name
        self.score = score