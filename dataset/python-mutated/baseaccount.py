class AccountManager:
    """I am responsible for managing a user's accounts.

    That is, remembering what accounts are available, their settings,
    adding and removal of accounts, etc.

    @ivar accounts: A collection of available accounts.
    @type accounts: mapping of strings to L{Account<interfaces.IAccount>}s.
    """

    def __init__(self):
        if False:
            return 10
        self.accounts = {}

    def getSnapShot(self):
        if False:
            i = 10
            return i + 15
        'A snapshot of all the accounts and their status.\n\n        @returns: A list of tuples, each of the form\n            (string:accountName, boolean:isOnline,\n            boolean:autoLogin, string:gatewayType)\n        '
        data = []
        for account in self.accounts.values():
            data.append((account.accountName, account.isOnline(), account.autoLogin, account.gatewayType))
        return data

    def isEmpty(self):
        if False:
            i = 10
            return i + 15
        return len(self.accounts) == 0

    def getConnectionInfo(self):
        if False:
            for i in range(10):
                print('nop')
        connectioninfo = []
        for account in self.accounts.values():
            connectioninfo.append(account.isOnline())
        return connectioninfo

    def addAccount(self, account):
        if False:
            i = 10
            return i + 15
        self.accounts[account.accountName] = account

    def delAccount(self, accountName):
        if False:
            i = 10
            return i + 15
        del self.accounts[accountName]

    def connect(self, accountName, chatui):
        if False:
            return 10
        '\n        @returntype: Deferred L{interfaces.IClient}\n        '
        return self.accounts[accountName].logOn(chatui)

    def disconnect(self, accountName):
        if False:
            for i in range(10):
                print('nop')
        pass

    def quit(self):
        if False:
            return 10
        pass