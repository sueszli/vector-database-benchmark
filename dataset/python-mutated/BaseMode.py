class BaseMode:

    def is_enabled(self, _query):
        if False:
            print('Hello World!')
        '\n        Return True if mode should be enabled for a query\n        '
        return False

    def on_query_change(self, _query):
        if False:
            i = 10
            return i + 15
        '\n        Triggered when user changes a search query\n        '

    def on_query_backspace(self, _query):
        if False:
            print('Hello World!')
        '\n        Return string to override default backspace and set the query to that string\n        '

    def handle_query(self, _query):
        if False:
            while True:
                i = 10
        '\n        :rtype: list of Results\n        '
        return []

    def get_triggers(self):
        if False:
            print('Hello World!')
        '\n        Returns an iterable of searchable results\n        '
        return []

    def get_fallback_results(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns a list of fallback results to\n        be displayed if nothing matches the user input\n        '
        return []