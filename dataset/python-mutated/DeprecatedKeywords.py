class DeprecatedKeywords:

    def deprecated_library_keyword(self):
        if False:
            for i in range(10):
                print('nop')
        '*DEPRECATED* Use keyword `Not Deprecated With Doc` instead!\n\n        Some more doc here. Ignore this in the warning.\n        '
        pass

    def deprecated_library_keyword_with_multiline_message(self):
        if False:
            return 10
        '*DEPRECATED* Multiline\n        message.\n\n        Some more doc here. Ignore this in the warning.\n        '
        pass

    def deprecated_library_keyword_without_extra_doc(self):
        if False:
            print('Hello World!')
        '*DEPRECATED*'
        pass

    def deprecated_library_keyword_with_stuff_to_ignore(self):
        if False:
            return 10
        '*DEPRECATED ignore this stuff*'
        pass

    def deprecated_keyword_returning(self):
        if False:
            while True:
                i = 10
        '*DEPRECATED.* But still returning a value!'
        return 42

    def not_deprecated_with_doc(self):
        if False:
            print('Hello World!')
        'Some Short Doc\n\n        Some more doc and ignore this *DEPRECATED*\n        '
        pass

    def not_deprecated_with_deprecated_prefix(self):
        if False:
            for i in range(10):
                print('nop')
        '*DEPRECATED ... just kidding!!'
        pass

    def not_deprecated_without_doc(self):
        if False:
            print('Hello World!')
        pass