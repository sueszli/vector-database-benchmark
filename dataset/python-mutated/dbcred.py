"""
Simple example of a db checker: define a L{ICredentialsChecker} implementation
that deals with a database backend to authenticate a user.
"""
from zope.interface import implementer
from twisted.cred import error
from twisted.cred.checkers import ICredentialsChecker
from twisted.cred.credentials import IUsernameHashedPassword, IUsernamePassword
from twisted.internet.defer import Deferred

@implementer(ICredentialsChecker)
class DBCredentialsChecker:
    """
    This class checks the credentials of incoming connections
    against a user table in a database.
    """

    def __init__(self, runQuery, query='SELECT username, password FROM user WHERE username = %s', customCheckFunc=None, caseSensitivePasswords=True):
        if False:
            for i in range(10):
                print('nop')
        "\n        @param runQuery: This will be called to get the info from the db.\n            Generally you'd want to create a\n            L{twisted.enterprice.adbapi.ConnectionPool} and pass it's runQuery\n            method here. Otherwise pass a function with the same prototype.\n        @type runQuery: C{callable}\n\n        @type query: query used to authenticate user.\n        @param query: C{str}\n\n        @param customCheckFunc: Use this if the passwords in the db are stored\n            as hashes. We'll just call this, so you can do the checking\n            yourself. It takes the following params:\n            (username, suppliedPass, dbPass) and must return a boolean.\n        @type customCheckFunc: C{callable}\n\n        @param caseSensitivePasswords: If true requires that every letter in\n            C{credentials.password} is exactly the same case as the it's\n            counterpart letter in the database.\n            This is only relevant if C{customCheckFunc} is not used.\n        @type caseSensitivePasswords: C{bool}\n        "
        self.runQuery = runQuery
        self.caseSensitivePasswords = caseSensitivePasswords
        self.customCheckFunc = customCheckFunc
        if customCheckFunc:
            self.credentialInterfaces = (IUsernamePassword,)
        else:
            self.credentialInterfaces = (IUsernamePassword, IUsernameHashedPassword)
        self.sql = query

    def requestAvatarId(self, credentials):
        if False:
            return 10
        '\n        Authenticates the kiosk against the database.\n        '
        for interface in self.credentialInterfaces:
            if interface.providedBy(credentials):
                break
        else:
            raise error.UnhandledCredentials()
        dbDeferred = self.runQuery(self.sql, (credentials.username,))
        deferred = Deferred()
        dbDeferred.addCallbacks(self._cbAuthenticate, self._ebAuthenticate, callbackArgs=(credentials, deferred), errbackArgs=(credentials, deferred))
        return deferred

    def _cbAuthenticate(self, result, credentials, deferred):
        if False:
            for i in range(10):
                print('nop')
        '\n        Checks to see if authentication was good. Called once the info has\n        been retrieved from the DB.\n        '
        if len(result) == 0:
            deferred.errback(error.UnauthorizedLogin('Username unknown'))
        else:
            (username, password) = result[0]
            if self.customCheckFunc:
                if self.customCheckFunc(username, credentials.password, password):
                    deferred.callback(credentials.username)
                else:
                    deferred.errback(error.UnauthorizedLogin('Password mismatch'))
            elif IUsernameHashedPassword.providedBy(credentials):
                if credentials.checkPassword(password):
                    deferred.callback(credentials.username)
                else:
                    deferred.errback(error.UnauthorizedLogin('Password mismatch'))
            elif IUsernamePassword.providedBy(credentials):
                if self.caseSensitivePasswords:
                    passOk = password.lower() == credentials.password.lower()
                else:
                    passOk = password == credentials.password
                if passOk:
                    deferred.callback(credentials.username)
                else:
                    deferred.errback(error.UnauthorizedLogin('Password mismatch'))
            else:
                deferred.errback(error.UnhandledCredentials())

    def _ebAuthenticate(self, message, credentials, deferred):
        if False:
            while True:
                i = 10
        '\n        The database lookup failed for some reason.\n        '
        deferred.errback(error.LoginFailed(message))

def main():
    if False:
        for i in range(10):
            print('nop')
    "\n    Run a simple echo pb server to test the checker. It defines a custom query\n    for dealing with sqlite special quoting, but otherwise it's a\n    straightforward use of the object.\n\n    You can test it running C{pbechoclient.py}.\n    "
    import sys
    from twisted.python import log
    log.startLogging(sys.stdout)
    import os
    if os.path.isfile('testcred'):
        os.remove('testcred')
    from twisted.enterprise import adbapi
    pool = adbapi.ConnectionPool('pysqlite2.dbapi2', 'testcred')
    query1 = 'CREATE TABLE user (\n            username string,\n            password string\n        )'
    query2 = "INSERT INTO user VALUES ('guest', 'guest')"

    def cb(res):
        if False:
            i = 10
            return i + 15
        pool.runQuery(query2)
    pool.runQuery(query1).addCallback(cb)
    checker = DBCredentialsChecker(pool.runQuery, query='SELECT username, password FROM user WHERE username = ?')
    import pbecho
    from twisted.cred.portal import Portal
    from twisted.spread import pb
    portal = Portal(pbecho.SimpleRealm())
    portal.registerChecker(checker)
    reactor.listenTCP(pb.portno, pb.PBServerFactory(portal))
if __name__ == '__main__':
    from twisted.internet import reactor
    reactor.callWhenRunning(main)
    reactor.run()