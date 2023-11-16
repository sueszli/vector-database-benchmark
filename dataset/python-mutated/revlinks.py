import re

class RevlinkMatch:

    def __init__(self, repo_urls, revlink):
        if False:
            i = 10
            return i + 15
        if isinstance(repo_urls, str):
            repo_urls = [repo_urls]
        self.repo_urls = [re.compile(url) for url in repo_urls]
        self.revlink = revlink

    def __call__(self, rev, repo):
        if False:
            for i in range(10):
                print('nop')
        for url in self.repo_urls:
            m = url.match(repo)
            if m:
                return m.expand(self.revlink) % rev
        return None
GithubRevlink = RevlinkMatch(repo_urls=['https://github.com/([^/]*)/([^/]*?)(?:\\.git)?$', 'git://github.com/([^/]*)/([^/]*?)(?:\\.git)?$', 'git@github.com:([^/]*)/([^/]*?)(?:\\.git)?$', 'ssh://git@github.com/([^/]*)/([^/]*?)(?:\\.git)?$'], revlink='https://github.com/\\1/\\2/commit/%s')
BitbucketRevlink = RevlinkMatch(repo_urls=['https://[^@]*@bitbucket.org/([^/]*)/([^/]*?)(?:\\.git)?$', 'git@bitbucket.org:([^/]*)/([^/]*?)(?:\\.git)?$'], revlink='https://bitbucket.org/\\1/\\2/commits/%s')

class GitwebMatch(RevlinkMatch):

    def __init__(self, repo_urls, revlink):
        if False:
            print('Hello World!')
        super().__init__(repo_urls=repo_urls, revlink=revlink + '?p=\\g<repo>;a=commit;h=%s')
SourceforgeGitRevlink = GitwebMatch(repo_urls=['^git://([^.]*).git.sourceforge.net/gitroot/(?P<repo>.*)$', '[^@]*@([^.]*).git.sourceforge.net:gitroot/(?P<repo>.*)$', 'ssh://(?:[^@]*@)?([^.]*).git.sourceforge.net/gitroot/(?P<repo>.*)$'], revlink='http://\\1.git.sourceforge.net/git/gitweb.cgi')
SourceforgeGitRevlink_AlluraPlatform = RevlinkMatch(repo_urls=['git://git.code.sf.net/p/(?P<repo>.*)$', 'http://git.code.sf.net/p/(?P<repo>.*)$', 'ssh://(?:[^@]*@)?git.code.sf.net/p/(?P<repo>.*)$'], revlink='https://sourceforge.net/p/\\1/ci/%s/')

class RevlinkMultiplexer:

    def __init__(self, *revlinks):
        if False:
            return 10
        self.revlinks = revlinks

    def __call__(self, rev, repo):
        if False:
            print('Hello World!')
        for revlink in self.revlinks:
            url = revlink(rev, repo)
            if url:
                return url
        return None
default_revlink_matcher = RevlinkMultiplexer(GithubRevlink, BitbucketRevlink, SourceforgeGitRevlink, SourceforgeGitRevlink_AlluraPlatform)