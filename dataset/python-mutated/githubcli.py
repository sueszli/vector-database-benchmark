from __future__ import unicode_literals
from __future__ import print_function
import click
from .github import GitHub
click.disable_unicode_literals_warning = True
pass_github = click.make_pass_decorator(GitHub)

class GitHubCli(object):
    """The GitHubCli, builds `click` commands and runs `GitHub` methods."""

    @click.group()
    @click.pass_context
    def cli(ctx):
        if False:
            i = 10
            return i + 15
        'Main entry point for GitHubCli.\n\n        :type ctx: :class:`click.core.Context`\n        :param ctx: An instance of click.core.Context that stores an instance\n            of `github.GitHub`.\n        '
        ctx.obj = GitHub()

    @cli.command()
    @click.option('-e', '--enterprise', is_flag=True)
    @pass_github
    def configure(github, enterprise):
        if False:
            for i in range(10):
                print('nop')
        "Configure gitsome.\n\n        Attempts to authenticate the user and to set up the user's news feed.\n\n        Usage/Example(s):\n            gh configure\n            gh configure -e\n            gh configure --enterprise\n\n        :type github: :class:`github.GitHub`\n        :param github: An instance of `github.GitHub`.\n        :type enterprise: bool\n        :param enterprise: Determines whether to configure GitHub Enterprise.\n            Default: False.\n        "
        github.configure(enterprise)

    @cli.command('create-comment')
    @click.argument('user_repo_number')
    @click.option('-t', '--text')
    @pass_github
    def create_comment(github, user_repo_number, text):
        if False:
            for i in range(10):
                print('nop')
        'Create a comment on the given issue.\n\n        Usage:\n            gh create-comment [user_repo_number] [-t/--text]\n\n        Example(s):\n            gh create-comment donnemartin/saws/1 -t "hello world"\n            gh create-comment donnemartin/saws/1 --text "hello world"\n\n        :type github: :class:`github.GitHub`\n        :param github: An instance of `github.GitHub`.\n\n        :type user_repo_number: str\n        :param user_repo_number: The user/repo/issue_number.\n\n        :type text: str\n        :param text: The comment text.\n        '
        github.create_comment(user_repo_number, text)

    @cli.command('create-issue')
    @click.argument('user_repo')
    @click.option('-t', '--issue_title')
    @click.option('-d', '--issue_desc', required=False)
    @pass_github
    def create_issue(github, user_repo, issue_title, issue_desc):
        if False:
            for i in range(10):
                print('nop')
        'Create an issue.\n\n        Usage:\n            gh create-issue [user_repo] [-t/--issue_title] [-d/--issue_desc]\n\n        Example(s):\n            gh create-issue donnemartin/gitsome -t "title"\n            gh create-issue donnemartin/gitsome -t "title" -d "desc"\n            gh create-issue donnemartin/gitsome --issue_title "title" --issue_desc "desc"  # NOQA\n\n        :type github: :class:`github.GitHub`\n        :param github: An instance of `github.GitHub`.\n\n        :type user_repo: str\n        :param user_repo: The user/repo.\n\n        :type issue_title: str\n        :param issue_title: The issue title.\n\n        :type issue_desc: str\n        :param issue_desc: The issue body (optional).\n        '
        github.create_issue(user_repo, issue_title, issue_desc)

    @cli.command('create-repo')
    @click.argument('repo_name')
    @click.option('-d', '--repo_desc', required=False)
    @click.option('-pr', '--private', is_flag=True)
    @pass_github
    def create_repo(github, repo_name, repo_desc, private):
        if False:
            print('Hello World!')
        'Create a repo.\n\n        Usage:\n            gh create-repo [repo_name] [-d/--repo_desc] [-pr/--private]\n\n        Example(s):\n            gh create-repo repo_name\n            gh create-repo repo_name -d "desc"\n            gh create-repo repo_name --repo_desc "desc"\n            gh create-repo repo_name -pr\n            gh create-repo repo_name --repo_desc "desc" --private\n\n        :type github: :class:`github.GitHub`\n        :param github: An instance of `github.GitHub`.\n\n        :type repo_name: str\n        :param repo_name: The repo name.\n\n        :type repo_desc: str\n        :param repo_desc: The repo description (optional).\n\n        :type private: bool\n        :param private: Determines whether the repo is private.  Default: False.\n        '
        github.create_repo(repo_name, repo_desc, private)

    @cli.command()
    @pass_github
    def emails(github):
        if False:
            for i in range(10):
                print('nop')
        "List all the user's registered emails.\n\n        Usage/Example(s):\n            gh emails\n\n        :type github: :class:`github.GitHub`\n        :param github: An instance of `github.GitHub`.\n        "
        github.emails()

    @cli.command()
    @click.option('-p', '--pager', is_flag=True)
    @pass_github
    def emojis(github, pager):
        if False:
            return 10
        'List all GitHub supported emojis.\n\n        Usage:\n            gh emojis [-p/--pager]\n\n        Example(s):\n            gh emojis | grep octo\n\n        :type github: :class:`github.GitHub`\n        :param github: An instance of `github.GitHub`.\n\n        :type pager: bool\n        :param pager: Determines whether to show the output in a pager,\n            if available.\n        '
        github.emojis(pager)

    @cli.command()
    @click.argument('user_or_repo', required=False, default='')
    @click.option('-pr', '--private', is_flag=True, default=False)
    @click.option('-p', '--pager', is_flag=True)
    @pass_github
    def feed(github, user_or_repo, private, pager):
        if False:
            while True:
                i = 10
        "List all activity for the given user or repo.\n\n        If `user_or_repo` is not provided, uses the logged in user's news feed\n        seen while visiting https://github.com.  If `user_or_repo` is provided,\n        shows either the public or `[-pr/--private]` feed activity of the user\n        or repo.\n\n        Usage:\n            gh feed [user_or_repo] [-pr/--private] [-p/--pager]\n\n        Examples:\n            gh feed\n            gh feed | grep foo\n            gh feed donnemartin\n            gh feed donnemartin -pr -p\n            gh feed donnemartin --private --pager\n            gh feed donnemartin/haxor-news -p\n\n        :type github: :class:`github.GitHub`\n        :param github: An instance of `github.GitHub`.\n\n        :type user_or_repo: str\n        :param user_or_repo: The user or repo to list events for (optional).\n            If no entry, defaults to the logged in user's feed.\n\n        :type private: bool\n        :param private: Determines whether to show the private events (True)\n            or public events (False).\n\n        :type pager: bool\n        :param pager: Determines whether to show the output in a pager,\n            if available.\n        "
        github.feed(user_or_repo, private, pager)

    @cli.command()
    @click.argument('user', required=False)
    @click.option('-p', '--pager', is_flag=True)
    @pass_github
    def followers(github, user, pager):
        if False:
            for i in range(10):
                print('nop')
        'List all followers and the total follower count.\n\n        Usage:\n            gh followers [user] [-p/--pager]\n\n        Example(s):\n            gh followers\n            gh followers -p\n            gh followers octocat --pager\n\n        :type github: :class:`github.GitHub`\n        :param github: An instance of `github.GitHub`.\n\n        :type user: str\n        :param user: The user login (optional).\n            If None, returns the followers of the logged in user.\n\n        :type pager: bool\n        :param pager: Determines whether to show the output in a pager,\n            if available.\n        '
        github.followers(user, pager)

    @cli.command()
    @click.argument('user', required=False)
    @click.option('-p', '--pager', is_flag=True)
    @pass_github
    def following(github, user, pager):
        if False:
            return 10
        'List all followed users and the total followed count.\n\n        Usage:\n            gh following [user] [-p/--pager]\n\n        Example(s):\n            gh following\n            gh following -p\n            gh following octocat --pager\n\n        :type github: :class:`github.GitHub`\n        :param github: An instance of `github.GitHub`.\n\n        :type user: str\n        :param user: The user login.\n            If None, returns the followed users of the logged in user.\n\n        :type pager: bool\n        :param pager: Determines whether to show the output in a pager,\n            if available.\n        '
        github.following(user, pager)

    @cli.command('gitignore-template')
    @click.argument('language')
    @pass_github
    def gitignore_template(github, language):
        if False:
            i = 10
            return i + 15
        'Output the gitignore template for the given language.\n\n        Usage:\n            gh gitignore-template [language]\n\n        Example(s):\n            gh gitignore-template Python\n            gh gitignore-template Python > .gitignore\n\n        :type github: :class:`github.GitHub`\n        :param github: An instance of `github.GitHub`.\n\n        :type language: str\n        :param language: The language.\n        '
        github.gitignore_template(language)

    @cli.command('gitignore-templates')
    @click.option('-p', '--pager', is_flag=True)
    @pass_github
    def gitignore_templates(github, pager):
        if False:
            i = 10
            return i + 15
        'Output all supported gitignore templates.\n\n        Usage:\n            gh gitignore-templates\n\n        Example(s):\n            gh gitignore-templates\n            gh gitignore-templates -p\n            gh gitignore-templates --pager\n\n        :type github: :class:`github.GitHub`\n        :param github: An instance of `github.GitHub`.\n\n        :type pager: bool\n        :param pager: Determines whether to show the output in a pager,\n            if available.\n        '
        github.gitignore_templates(pager)

    @cli.command()
    @click.argument('user_repo_number')
    @pass_github
    def issue(github, user_repo_number):
        if False:
            i = 10
            return i + 15
        'Output detailed information about the given issue.\n\n        Usage:\n            gh issue [user_repo_number]\n\n        Example(s):\n            gh issue donnemartin/saws/1\n\n        :type github: :class:`github.GitHub`\n        :param github: An instance of `github.GitHub`.\n\n        :type user_repo_number: str\n        :param user_repo_number: The user/repo/issue_number.\n        '
        github.issue(user_repo_number)

    @cli.command()
    @click.option('-f', '--issue_filter', required=False, default='subscribed')
    @click.option('-s', '--issue_state', required=False, default='open')
    @click.option('-l', '--limit', required=False, default=1000)
    @click.option('-p', '--pager', is_flag=True)
    @pass_github
    def issues(github, issue_filter, issue_state, limit, pager):
        if False:
            while True:
                i = 10
        'List all issues matching the filter.\n\n        Usage:\n            gh issues [-f/--issue_filter] [-s/--issue_state] [-l/--limit] [-p/--pager]  # NOQA\n\n        Example(s):\n            gh issues\n            gh issues -f assigned\n            gh issues ---issue_filter created\n            gh issues -s all -l 20 -p\n            gh issues --issue_state closed --limit 20 --pager\n            gh issues -f created -s all -p\n\n        :type github: :class:`github.GitHub`\n        :param github: An instance of `github.GitHub`.\n\n        :type issue_filter: str\n        :param issue_filter: assigned, created, mentioned, subscribed (default).\n\n        :type issue_state: str\n        :param issue_state: all, open (default), closed.\n\n        :type limit: int\n        :param limit: The number of items to display.\n\n        :type pager: bool\n        :param pager: Determines whether to show the output in a pager,\n            if available.\n        '
        github.issues_setup(issue_filter, issue_state, limit, pager)

    @cli.command()
    @click.argument('license_name')
    @pass_github
    def license(github, license_name):
        if False:
            for i in range(10):
                print('nop')
        'Output the license template for the given license.\n\n        Usage:\n            gh license [license_name]\n\n        Example(s):\n            gh license apache-2.0\n            gh license mit > LICENSE\n\n        :type github: :class:`github.GitHub`\n        :param github: An instance of `github.GitHub`.\n\n        :type license_name: str\n        :param license_name: The license name.\n        '
        github.license(license_name)

    @cli.command()
    @pass_github
    def licenses(github):
        if False:
            for i in range(10):
                print('nop')
        'Output all supported license templates.\n\n        Usage/Example(s):\n            gh licenses\n\n        :type github: :class:`github.GitHub`\n        :param github: An instance of `github.GitHub`.\n        '
        github.licenses()

    @cli.command()
    @click.option('-b', '--browser', is_flag=True)
    @click.option('-t', '--text_avatar', is_flag=True)
    @click.option('-l', '--limit', required=False, default=1000)
    @click.option('-p', '--pager', is_flag=True)
    @pass_github
    def me(github, browser, text_avatar, limit, pager):
        if False:
            i = 10
            return i + 15
        'List information about the logged in user.\n\n        Usage:\n            gh me [-b/--browser] [-t/--text_avatar] [-l/--limit] [-p/--pager]\n\n        Example(s):\n            gh me\n            gh me -b\n            gh me --browser\n            gh me -t -l 20 -p\n            gh me --text_avatar --limit 20 --pager\n\n        :type github: :class:`github.GitHub`\n        :param github: An instance of `github.GitHub`.\n\n        :type browser: bool\n        :param browser: Determines whether to view the profile\n            in a browser, or in the terminal.\n\n        :type text_avatar: bool\n        :param text_avatar: Determines whether to view the profile\n            avatar in plain text instead of ansi (default).\n            On Windows this value is always set to True due to lack of\n            support of `img2txt` on Windows.\n\n        :type limit: int\n        :param limit: The number of user repos to display.\n\n        :type pager: bool\n        :param pager: Determines whether to show the output in a pager,\n            if available.\n        '
        github.user_me(browser, text_avatar, limit, pager)

    @cli.command()
    @click.option('-l', '--limit', required=False, default=1000)
    @click.option('-p', '--pager', is_flag=True)
    @pass_github
    def notifications(github, limit, pager):
        if False:
            for i in range(10):
                print('nop')
        'List all notifications.\n\n        Usage:\n            gh notifications [-l/--limit] [-p/--pager]\n\n        Example(s):\n            gh notifications\n            gh notifications -l 20 -p\n            gh notifications --limit 20 --pager\n\n        :type github: :class:`github.GitHub`\n        :param github: An instance of `github.GitHub`.\n\n        :type limit: int\n        :param limit: The number of items to display.\n\n        :type pager: bool\n        :param pager: Determines whether to show the output in a pager,\n            if available.\n        '
        github.notifications(limit, pager)

    @cli.command('octo')
    @click.argument('say', required=False)
    @pass_github
    def octocat(github, say):
        if False:
            while True:
                i = 10
        'Output an Easter egg or the given message from Octocat.\n\n        Usage:\n            gh octo [say]\n\n        Example(s):\n            gh octo\n            gh octo "foo bar"\n\n        :type github: :class:`github.GitHub`\n        :param github: An instance of `github.GitHub`.\n\n        :type say: str\n        :param say: What Octocat should say.\n                If say is None, octocat speaks an Easter egg.\n        '
        github.octocat(say)

    @cli.command('pull-request')
    @click.argument('user_repo_number')
    @pass_github
    def pull_request(github, user_repo_number):
        if False:
            i = 10
            return i + 15
        'Output detailed information about the given pull request.\n\n        Usage:\n            gh pull-request [user_repo_number]\n\n        Example(s):\n            gh pull-request donnemartin/saws/80\n\n        :type github: :class:`github.GitHub`\n        :param github: An instance of `github.GitHub`.\n\n        :type user_repo_number: str\n        :param user_repo_number: The user/repo/pull_number.\n        '
        github.issue(user_repo_number)

    @cli.command('pull-requests')
    @click.option('-l', '--limit', required=False, default=1000)
    @click.option('-p', '--pager', is_flag=True)
    @pass_github
    def pull_requests(github, limit, pager):
        if False:
            while True:
                i = 10
        'List all pull requests.\n\n        Usage:\n            gh pull-requests [-l/--limit] [-p/--pager]\n\n        Example(s):\n            gh pull-requests\n            gh pull-requests -l 20 -p\n            gh pull-requests --limit 20 --pager\n\n        :type github: :class:`github.GitHub`\n        :param github: An instance of `github.GitHub`.\n\n        :type limit: int\n        :param limit: The number of items to display.\n\n        :type pager: bool\n        :param pager: Determines whether to show the output in a pager,\n            if available.\n        '
        github.pull_requests(limit, pager)

    @cli.command('rate-limit')
    @pass_github
    def rate_limit(github):
        if False:
            for i in range(10):
                print('nop')
        'Output the rate limit.  Not available for GitHub Enterprise.\n\n        Logged in users can make 5000 requests per hour.\n        See: https://developer.github.com/v3/#rate-limiting\n\n        Usage/Example(s):\n            gh rate-limit\n\n        :type github: :class:`github.GitHub`\n        :param github: An instance of `github.GitHub`.\n        '
        github.rate_limit()

    @cli.command('repos')
    @click.argument('repo_filter', required=False, default='')
    @click.option('-l', '--limit', required=False, default=1000)
    @click.option('-p', '--pager', is_flag=True)
    @pass_github
    def repositories(github, repo_filter, limit, pager):
        if False:
            for i in range(10):
                print('nop')
        'List all repos matching the given filter.\n\n        Usage:\n            gh repos [repo_filter] [-l/--limit] [-p/--pager]\n\n        Example(s):\n            gh repos\n            gh repos "data-science"\n            gh repos "data-science" -l 20 -p\n            gh repos "data-science" --limit 20 --pager\n\n        :type github: :class:`github.GitHub`\n        :param github: An instance of `github.GitHub`.\n\n        :type repo_filter: str\n        :param repo_filter:  The filter for repo names.\n            Only repos matching the filter will be returned.\n            If None, outputs all the logged in user\'s repos.\n\n        :type limit: int\n        :param limit: The number of items to display.\n\n        :type pager: bool\n        :param pager: Determines whether to show the output in a pager,\n            if available.\n        '
        github.repositories_setup(repo_filter, limit, pager)

    @cli.command('repo')
    @click.argument('user_repo')
    @pass_github
    def repository(github, user_repo):
        if False:
            while True:
                i = 10
        'Output detailed information about the given repo.\n\n        Usage:\n            gh repo [user_repo]\n\n        Example(s):\n            gh repo donnemartin/gitsome\n\n        :type github: :class:`github.GitHub`\n        :param github: An instance of `github.GitHub`.\n\n        :type user_repo: str\n        :param user_repo: The user/repo.\n        '
        github.repository(user_repo)

    @cli.command('search-issues')
    @click.argument('query')
    @click.option('-l', '--limit', required=False, default=1000)
    @click.option('-p', '--pager', is_flag=True)
    @pass_github
    def search_issues(github, query, limit, pager):
        if False:
            print('Hello World!')
        'Search for all issues matching the given query.\n\n        Usage:\n            gh search-issues [query] [-l/--limit] [-p/--pager]\n\n        Example(s):\n            gh search-issues "foo type:pr author:donnemartin" -l 20 -p\n            gh search-issues "foobarbaz in:title created:>=2015-01-01" --limit 20 --pager  # NOQA\n\n        Additional Example(s):\n            Search issues that have your user name tagged @donnemartin:\n                gh search-issues "is:issue donnemartin is:open" -p\n\n            Search issues that have the most +1s:\n                gh search-issues "is:open is:issue sort:reactions-+1-desc" -p\n\n            Search issues that have the most comments:\n                gh search-issues "is:open is:issue sort:comments-desc" -p\n\n            Search issues with the "help wanted" tag:\n                gh search-issues "is:open is:issue label:"help wanted"" -p\n\n            Search all your open private issues:\n                gh search-issues "is:open is:issue is:private" -p\n\n        The query can contain any combination of the following supported\n        qualifers:\n\n        - `type` With this qualifier you can restrict the search to issues\n          or pull request only.\n        - `in` Qualifies which fields are searched. With this qualifier you\n          can restrict the search to just the title, body, comments, or any\n          combination of these.\n        - `author` Finds issues created by a certain user.\n        - `assignee` Finds issues that are assigned to a certain user.\n        - `mentions` Finds issues that mention a certain user.\n        - `commenter` Finds issues that a certain user commented on.\n        - `involves` Finds issues that were either created by a certain user,\n          assigned to that user, mention that user, or were commented on by\n          that user.\n        - `state` Filter issues based on whether they’re open or closed.\n        - `labels` Filters issues based on their labels.\n        - `language` Searches for issues within repositories that match a\n          certain language.\n        - `created` or `updated` Filters issues based on times of creation,\n          or when they were last updated.\n        - `comments` Filters issues based on the quantity of comments.\n        - `user` or `repo` Limits searches to a specific user or\n          repository.\n\n        For more information about these qualifiers, see: http://git.io/d1oELA\n\n        :type github: :class:`github.GitHub`\n        :param github: An instance of `github.GitHub`.\n\n        :type query: str\n        :param query: The search query.\n\n        :type limit: int\n        :param limit: The number of items to display.\n\n        :type pager: bool\n        :param pager: Determines whether to show the output in a pager,\n            if available.\n        '
        github.search_issues(query, limit, pager)

    @cli.command('search-repos')
    @click.argument('query')
    @click.option('-s', '--sort', required=False, default='')
    @click.option('-l', '--limit', required=False, default=1000)
    @click.option('-p', '--pager', is_flag=True)
    @pass_github
    def search_repositories(github, query, sort, limit, pager):
        if False:
            print('Hello World!')
        'Search for all repos matching the given query.\n\n        Usage:\n            gh search-repos [query] [-s/--sort] [-l/--limit] [-p/--pager]\n\n        Example(s):\n            gh search-repos "maps language:python" -s stars -l 20 -p\n            gh search-repos "created:>=2015-01-01 stars:>=1000 language:python" --sort stars --limit 20 --pager  # NOQA\n\n        The query can contain any combination of the following supported\n        qualifers:\n\n        - `in` Qualifies which fields are searched. With this qualifier you\n          can restrict the search to just the repository name, description,\n          readme, or any combination of these.\n        - `size` Finds repositories that match a certain size (in\n          kilobytes).\n        - `forks` Filters repositories based on the number of forks, and/or\n          whether forked repositories should be included in the results at\n          all.\n        - `created` or `pushed` Filters repositories based on times of\n          creation, or when they were last updated. Format: `YYYY-MM-DD`.\n          Examples: `created:<2011`, `pushed:<2013-02`,\n          `pushed:>=2013-03-06`\n        - `user` or `repo` Limits searches to a specific user or\n          repository.\n        - `language` Searches repositories based on the language they\'re\n          written in.\n        - `stars` Searches repositories based on the number of stars.\n\n        For more information about these qualifiers, see: http://git.io/4Z8AkA\n\n        :type github: :class:`github.GitHub`\n        :param github: An instance of `github.GitHub`.\n\n        :type query: str\n        :param query: The search query.\n\n        :type sort: str\n        :param sort: Optional: \'stars\', \'forks\', \'updated\'.\n            If not specified, sorting is done by query best match.\n\n        :type limit: int\n        :param limit: The number of items to display.\n\n        :type pager: bool\n        :param pager: Determines whether to show the output in a pager,\n            if available.\n        '
        github.search_repositories(query, sort, limit, pager)

    @cli.command()
    @click.argument('repo_filter', required=False, default='')
    @click.option('-l', '--limit', required=False, default=1000)
    @click.option('-p', '--pager', is_flag=True)
    @pass_github
    def starred(github, repo_filter, limit, pager):
        if False:
            i = 10
            return i + 15
        'Output starred repos.\n\n        Usage:\n            gh starred [repo_filter] [-l/--limit] [-p/--pager]\n\n        Example(s):\n            gh starred\n            gh starred foo -l 20 -p\n            gh starred foo --limit 20 --pager\n\n        :type github: :class:`github.GitHub`\n        :param github: An instance of `github.GitHub`.\n\n        :type repo_filter: str\n        :param repo_filter:  The filter for repo names.\n            Only repos matching the filter will be returned.\n            If None, outputs all starred repos.\n\n        :type limit: int\n        :param limit: The number of items to display.\n\n        :type pager: bool\n        :param pager: Determines whether to show the output in a pager,\n            if available.\n        '
        github.starred(repo_filter, limit, pager)

    @cli.command()
    @click.argument('language', required=False, default='Overall')
    @click.option('-w', '--weekly', is_flag=True)
    @click.option('-m', '--monthly', is_flag=True)
    @click.option('-D', '--devs', is_flag=True)
    @click.option('-b', '--browser', is_flag=True)
    @click.option('-p', '--pager', is_flag=True)
    @pass_github
    def trending(github, language, weekly, monthly, devs, browser, pager):
        if False:
            for i in range(10):
                print('nop')
        "List trending repos for the given language.\n\n        Usage:\n            gh trending [language] [-w/--weekly] [-m/--monthly] [-D/--devs] [-b/--browser] [-p/--pager]  # NOQA\n\n        Example(s):\n            gh trending\n            gh trending Python -w -p\n            gh trending Python --weekly --devs --browser\n            gh trending --browser\n\n        :type github: :class:`github.GitHub`\n        :param github: An instance of `github.GitHub`.\n\n        :type language: str\n        :param language: The language (optional).\n            If blank, shows 'Overall'.\n\n        :type weekly: bool\n        :param weekly: Determines whether to show the weekly rankings.\n            Daily is the default.\n\n        :type monthly: bool\n        :param monthly: Determines whether to show the monthly rankings.\n            Daily is the default.\n            If both `monthly` and `weekly` are set, `monthly` takes precedence.\n\n        :type devs: bool\n        :param devs: determines whether to display the trending\n                devs or repos.  Only valid with the -b/--browser option.\n\n        :type browser: bool\n        :param browser: Determines whether to view the profile\n                in a browser, or in the terminal.\n\n        :type pager: bool\n        :param pager: Determines whether to show the output in a pager,\n            if available.\n        "
        github.trending(language, weekly, monthly, devs, browser, pager)

    @cli.command()
    @click.argument('user_id', required=True)
    @click.option('-b', '--browser', is_flag=True)
    @click.option('-t', '--text_avatar', is_flag=True)
    @click.option('-l', '--limit', required=False, default=1000)
    @click.option('-p', '--pager', is_flag=True)
    @pass_github
    def user(github, user_id, browser, text_avatar, limit, pager):
        if False:
            i = 10
            return i + 15
        'List information about the given user.\n\n        Usage:\n            gh user [user_id] [-b/--browser] [-t/--text_avatar] [-l/--limit] [-p/--pager]  # NOQA\n\n        Example(s):\n            gh user octocat\n            gh user octocat -b\n            gh user octocat --browser\n            gh user octocat -t -l 10 -p\n            gh user octocat --text_avatar --limit 10 --pager\n\n        :type github: :class:`github.GitHub`\n        :param github: An instance of `github.GitHub`.\n\n        :type user_id: str\n        :param user_id: The user id/login.\n            If None, returns followers of the logged in user.\n\n        :type browser: bool\n        :param browser: Determines whether to view the profile\n            in a browser, or in the terminal.\n\n        :type text_avatar: bool\n        :param text_avatar: Determines whether to view the profile\n            avatar in plain text instead of ansi (default).\n            On Windows this value is always set to True due to lack of\n            support of `img2txt` on Windows.\n\n        :type limit: int\n        :param limit: The number of items to display.\n\n        :type pager: bool\n        :param pager: Determines whether to show the output in a pager,\n            if available.\n        '
        github.user(user_id, browser, text_avatar, limit, pager)

    @cli.command()
    @click.argument('index')
    @click.option('-b', '--browser', is_flag=True)
    @pass_github
    def view(github, index, browser):
        if False:
            for i in range(10):
                print('nop')
        'View the given notification/repo/issue/pull_request/user index.\n\n        This method is meant to be called after one of the following commands\n        which outputs a table of notifications/repos/issues/pull_requests/users:\n\n            gh repos\n            gh search_repos\n            gh starred\n\n            gh issues\n            gh pull_requests\n            gh search_issues\n\n            gh notifications\n            gh trending\n\n            gh user\n            gh me\n\n        Usage:\n            gh view [index] [-b/--browser]\n\n        Example(s):\n            gh repos\n            gh view 1\n\n            gh starred\n            gh view 1 -b\n            gh view 1 --browser\n\n        :type github: :class:`github.GitHub`\n        :param github: An instance of `github.GitHub`.\n\n        :type index: str\n        :param index: Determines the index to view.\n\n        :type browser: bool\n        :param browser: Determines whether to view the profile\n            in a browser, or in the terminal.\n        '
        github.view(int(index), browser)