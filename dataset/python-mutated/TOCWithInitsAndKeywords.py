class TOCWithInitsAndKeywords:
    """
    = First entry =

    TOC in somewhat strange place.

        %TOC%

    = Second =

             = 3 =

    %TOC% not replaced here
    """

    def __init__(self, arg=True):
        if False:
            while True:
                i = 10
        pass

    def keyword(self):
        if False:
            return 10
        'Tags: tag'
        pass