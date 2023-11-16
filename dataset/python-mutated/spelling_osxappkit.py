"""
AppKit Spelling: Implements spelling backend based on OSX's spellchecking
                 features provided by the ApplicationKit.

                 NOTE:
                    Requires pyobjc and setuptools to be installed!
                    `sudo easy_install pyobjc setuptools`

                 Developers should read:
                    http://developer.apple.com/mac/library/documentation/
                        Cocoa/Conceptual/SpellCheck/SpellCheck.html
                    http://developer.apple.com/cocoa/pyobjc.html
"""
from AppKit import NSSpellChecker, NSMakeRange
from kivy.core.spelling import SpellingBase, NoSuchLangError

class SpellingOSXAppKit(SpellingBase):
    """
    Spelling backend based on OSX's spelling features provided by AppKit.
    """

    def __init__(self, language=None):
        if False:
            return 10
        self._language = NSSpellChecker.alloc().init()
        super(SpellingOSXAppKit, self).__init__(language)

    def select_language(self, language):
        if False:
            return 10
        success = self._language.setLanguage_(language)
        if not success:
            err = 'AppKit Backend: No language "%s" ' % (language,)
            raise NoSuchLangError(err)

    def list_languages(self):
        if False:
            print('Hello World!')
        return list(self._language.availableLanguages())

    def check(self, word):
        if False:
            while True:
                i = 10
        if not word:
            return None
        err = 'check() not currently supported by the OSX AppKit backend'
        raise NotImplementedError(err)

    def suggest(self, fragment):
        if False:
            print('Hello World!')
        l = self._language
        try:
            return list(l.guessesForWord_(fragment))
        except AttributeError:
            checkrange = NSMakeRange(0, len(fragment))
            g = l.guessesForWordRange_inString_language_inSpellDocumentWithTag_(checkrange, fragment, l.language(), 0)
            return list(g)