"""Sherlock Notify Module

This module defines the objects for notifying the caller about the
results of queries.
"""
from result import QueryStatus
from colorama import Fore, Style
import webbrowser
globvar = 0

class QueryNotify:
    """Query Notify Object.

    Base class that describes methods available to notify the results of
    a query.
    It is intended that other classes inherit from this base class and
    override the methods to implement specific functionality.
    """

    def __init__(self, result=None):
        if False:
            print('Hello World!')
        'Create Query Notify Object.\n\n        Contains information about a specific method of notifying the results\n        of a query.\n\n        Keyword Arguments:\n        self                   -- This object.\n        result                 -- Object of type QueryResult() containing\n                                  results for this query.\n\n        Return Value:\n        Nothing.\n        '
        self.result = result

    def start(self, message=None):
        if False:
            i = 10
            return i + 15
        'Notify Start.\n\n        Notify method for start of query.  This method will be called before\n        any queries are performed.  This method will typically be\n        overridden by higher level classes that will inherit from it.\n\n        Keyword Arguments:\n        self                   -- This object.\n        message                -- Object that is used to give context to start\n                                  of query.\n                                  Default is None.\n\n        Return Value:\n        Nothing.\n        '

    def update(self, result):
        if False:
            while True:
                i = 10
        'Notify Update.\n\n        Notify method for query result.  This method will typically be\n        overridden by higher level classes that will inherit from it.\n\n        Keyword Arguments:\n        self                   -- This object.\n        result                 -- Object of type QueryResult() containing\n                                  results for this query.\n\n        Return Value:\n        Nothing.\n        '
        self.result = result

    def finish(self, message=None):
        if False:
            print('Hello World!')
        'Notify Finish.\n\n        Notify method for finish of query.  This method will be called after\n        all queries have been performed.  This method will typically be\n        overridden by higher level classes that will inherit from it.\n\n        Keyword Arguments:\n        self                   -- This object.\n        message                -- Object that is used to give context to start\n                                  of query.\n                                  Default is None.\n\n        Return Value:\n        Nothing.\n        '

    def __str__(self):
        if False:
            for i in range(10):
                print('nop')
        'Convert Object To String.\n\n        Keyword Arguments:\n        self                   -- This object.\n\n        Return Value:\n        Nicely formatted string to get information about this object.\n        '
        return str(self.result)

class QueryNotifyPrint(QueryNotify):
    """Query Notify Print Object.

    Query notify class that prints results.
    """

    def __init__(self, result=None, verbose=False, print_all=False, browse=False):
        if False:
            i = 10
            return i + 15
        'Create Query Notify Print Object.\n\n        Contains information about a specific method of notifying the results\n        of a query.\n\n        Keyword Arguments:\n        self                   -- This object.\n        result                 -- Object of type QueryResult() containing\n                                  results for this query.\n        verbose                -- Boolean indicating whether to give verbose output.\n        print_all              -- Boolean indicating whether to only print all sites, including not found.\n        browse                 -- Boolean indicating whether to open found sites in a web browser.\n\n        Return Value:\n        Nothing.\n        '
        super().__init__(result)
        self.verbose = verbose
        self.print_all = print_all
        self.browse = browse
        return

    def start(self, message):
        if False:
            for i in range(10):
                print('nop')
        'Notify Start.\n\n        Will print the title to the standard output.\n\n        Keyword Arguments:\n        self                   -- This object.\n        message                -- String containing username that the series\n                                  of queries are about.\n\n        Return Value:\n        Nothing.\n        '
        title = 'Checking username'
        print(Style.BRIGHT + Fore.GREEN + '[' + Fore.YELLOW + '*' + Fore.GREEN + f'] {title}' + Fore.WHITE + f' {message}' + Fore.GREEN + ' on:')
        print('\r')
        return

    def countResults(self):
        if False:
            print('Hello World!')
        'This function counts the number of results. Every time the function is called,\n        the number of results is increasing.\n\n        Keyword Arguments:\n        self                   -- This object.\n\n        Return Value:\n        The number of results by the time we call the function.\n        '
        global globvar
        globvar += 1
        return globvar

    def update(self, result):
        if False:
            print('Hello World!')
        'Notify Update.\n\n        Will print the query result to the standard output.\n\n        Keyword Arguments:\n        self                   -- This object.\n        result                 -- Object of type QueryResult() containing\n                                  results for this query.\n\n        Return Value:\n        Nothing.\n        '
        self.result = result
        response_time_text = ''
        if self.result.query_time is not None and self.verbose is True:
            response_time_text = f' [{round(self.result.query_time * 1000)}ms]'
        if result.status == QueryStatus.CLAIMED:
            self.countResults()
            print(Style.BRIGHT + Fore.WHITE + '[' + Fore.GREEN + '+' + Fore.WHITE + ']' + response_time_text + Fore.GREEN + f' {self.result.site_name}: ' + Style.RESET_ALL + f'{self.result.site_url_user}')
            if self.browse:
                webbrowser.open(self.result.site_url_user, 2)
        elif result.status == QueryStatus.AVAILABLE:
            if self.print_all:
                print(Style.BRIGHT + Fore.WHITE + '[' + Fore.RED + '-' + Fore.WHITE + ']' + response_time_text + Fore.GREEN + f' {self.result.site_name}:' + Fore.YELLOW + ' Not Found!')
        elif result.status == QueryStatus.UNKNOWN:
            if self.print_all:
                print(Style.BRIGHT + Fore.WHITE + '[' + Fore.RED + '-' + Fore.WHITE + ']' + Fore.GREEN + f' {self.result.site_name}:' + Fore.RED + f' {self.result.context}' + Fore.YELLOW + ' ')
        elif result.status == QueryStatus.ILLEGAL:
            if self.print_all:
                msg = 'Illegal Username Format For This Site!'
                print(Style.BRIGHT + Fore.WHITE + '[' + Fore.RED + '-' + Fore.WHITE + ']' + Fore.GREEN + f' {self.result.site_name}:' + Fore.YELLOW + f' {msg}')
        else:
            raise ValueError(f"Unknown Query Status '{result.status}' for site '{self.result.site_name}'")
        return

    def finish(self, message='The processing has been finished.'):
        if False:
            for i in range(10):
                print('nop')
        'Notify Start.\n        Will print the last line to the standard output.\n        Keyword Arguments:\n        self                   -- This object.\n        message                -- The 2 last phrases.\n        Return Value:\n        Nothing.\n        '
        NumberOfResults = self.countResults() - 1
        print(Style.BRIGHT + Fore.GREEN + '[' + Fore.YELLOW + '*' + Fore.GREEN + '] Search completed with' + Fore.WHITE + f' {NumberOfResults} ' + Fore.GREEN + 'results' + Style.RESET_ALL)

    def __str__(self):
        if False:
            for i in range(10):
                print('nop')
        'Convert Object To String.\n\n        Keyword Arguments:\n        self                   -- This object.\n\n        Return Value:\n        Nicely formatted string to get information about this object.\n        '
        return str(self.result)