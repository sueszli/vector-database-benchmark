import string

class CookieCleaner:
    """This class cleans cookies we haven't seen before.  The basic idea is to
    kill sessions, which isn't entirely straight-forward.  Since we want this to
    be generalized, there's no way for us to know exactly what cookie we're trying
    to kill, which also means we don't know what domain or path it has been set for.

    The rule with cookies is that specific overrides general.  So cookies that are
    set for mail.foo.com override cookies with the same name that are set for .foo.com,
    just as cookies that are set for foo.com/mail override cookies with the same name
    that are set for foo.com/

    The best we can do is guess, so we just try to cover our bases by expiring cookies
    in a few different ways.  The most obvious thing to do is look for individual cookies
    and nail the ones we haven't seen coming from the server, but the problem is that cookies are often
    set by Javascript instead of a Set-Cookie header, and if we block those the site
    will think cookies are disabled in the browser.  So we do the expirations and whitlisting
    based on client,server tuples.  The first time a client hits a server, we kill whatever
    cookies we see then.  After that, we just let them through.  Not perfect, but pretty effective.

    """
    _instance = None

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.cleanedCookies = set()
        self.enabled = False

    @staticmethod
    def getInstance():
        if False:
            for i in range(10):
                print('nop')
        if CookieCleaner._instance == None:
            CookieCleaner._instance = CookieCleaner()
        return CookieCleaner._instance

    def setEnabled(self, enabled):
        if False:
            for i in range(10):
                print('nop')
        self.enabled = enabled

    def isClean(self, method, client, host, headers):
        if False:
            return 10
        if method == 'POST':
            return True
        if not self.enabled:
            return True
        if not self.hasCookies(headers):
            return True
        return (client, self.getDomainFor(host)) in self.cleanedCookies

    def getExpireHeaders(self, method, client, host, headers, path):
        if False:
            print('Hello World!')
        domain = self.getDomainFor(host)
        self.cleanedCookies.add((client, domain))
        expireHeaders = []
        for cookie in headers['cookie'].split(';'):
            cookie = cookie.split('=')[0].strip()
            expireHeadersForCookie = self.getExpireCookieStringFor(cookie, host, domain, path)
            expireHeaders.extend(expireHeadersForCookie)
        return expireHeaders

    def hasCookies(self, headers):
        if False:
            for i in range(10):
                print('nop')
        return 'cookie' in headers

    def getDomainFor(self, host):
        if False:
            while True:
                i = 10
        hostParts = host.split('.')
        return '.' + hostParts[-2] + '.' + hostParts[-1]

    def getExpireCookieStringFor(self, cookie, host, domain, path):
        if False:
            return 10
        pathList = path.split('/')
        expireStrings = list()
        expireStrings.append(cookie + '=' + 'EXPIRED;Path=/;Domain=' + domain + ';Expires=Mon, 01-Jan-1990 00:00:00 GMT\r\n')
        expireStrings.append(cookie + '=' + 'EXPIRED;Path=/;Domain=' + host + ';Expires=Mon, 01-Jan-1990 00:00:00 GMT\r\n')
        if len(pathList) > 2:
            expireStrings.append(cookie + '=' + 'EXPIRED;Path=/' + pathList[1] + ';Domain=' + domain + ';Expires=Mon, 01-Jan-1990 00:00:00 GMT\r\n')
            expireStrings.append(cookie + '=' + 'EXPIRED;Path=/' + pathList[1] + ';Domain=' + host + ';Expires=Mon, 01-Jan-1990 00:00:00 GMT\r\n')
        return expireStrings