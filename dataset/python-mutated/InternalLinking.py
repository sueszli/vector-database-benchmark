class InternalLinking:
    u"""Library for testing libdoc's internal linking.

    = Linking to sections =

    - `Introduction`
    - `Library INTROduction`
    - `importing`
    - `Library Importing`
    - `ShortCuts`
    - `Keywords`

    = Linking to keywords =

    - `Keyword`
    - `secoNd kEywoRD`

    = Linking to headers =

    == First level header can be linked ==

    - `linking to headers`
    - `first = level =`

    == Other levels can be linked too ==

    - `Second level`
    - `third level`

    =   First = Level = =

    == Second level ==

    === Third level ===

    = Escaping =

    == Percent encoding: !"#%/()=?|+-_.!~*'() ==

    == HTML entities: &<> ==

    == Non-ASCII: ä☃ ==

    = Formatting =

    Non-matching `backticks` just get special formatting.
    """

    def __init__(self, argument=None):
        if False:
            while True:
                i = 10
        'Importing. See `introduction`, `formatting` and `keyword` for details.'

    def keyword(self):
        if False:
            i = 10
            return i + 15
        'First keyword here. See also `Importing` and `Second Keyword`.'

    def second_keyword(self, arg):
        if False:
            return 10
        'We got `arg`. And have `no link`. Except to `Second LEVEL`.\n\n        = Not linkable =\n\n        We are `linking to headers` and `shortcuts` but not to `not linkable`.\n        '

    def escaping(self):
        if False:
            while True:
                i = 10
        u'Escaped links:\n        - `Percent encoding: !"#%/()=?|+-_.!~*\'()`\n        - `HTML entities: &<>`\n        - `Non-ASCII: ä☃`\n        '