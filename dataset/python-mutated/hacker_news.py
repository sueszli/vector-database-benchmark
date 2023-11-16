from __future__ import print_function
from __future__ import division
import platform
import re
import sys
import webbrowser
import click
from .compat import HTMLParser
from .compat import urlparse
from .config import Config
from .lib.haxor.haxor import HackerNewsApi, HTTPError, InvalidItemID, InvalidUserID
from .lib.pretty_date_time import pretty_date_time
from .onions import onions
from .web_viewer import WebViewer

class HackerNews(object):
    """Encapsulate Hacker News.

        :type COMMENT_INDENT: str (const)
        :param COMMENT_INDENT: The comment indent.

        :type COMMENT_UNSEEN: str (const)
        :param COMMENT_UNSEEN: The adornment for unseen
            comments.

        :type config: :class:`config.Config`
        :param config: An instance of `config.Config`.

        :type html: :class:`HTMLParser`
        :param html: An instance of `HTMLParser`.

        :type MAX_LIST_INDEX: int (const)
        :param MAX_LIST_INDEX: The maximum 1-based index value
            hn view will use to match item_ids.  Any value larger than
            MAX_LIST_INDEX will result in hn view treating that index as an
            actual post id.

        :type MAX_SNIPPET_LENGTH: int (const)
        :param MAX_SNIPPET_LENGTH: The max length of a comment snippet shown
            when filtering comments.

        :type hacker_news_api: :class:`haxor.HackerNewsApi`
        :param hacker_news_api: An instance of `haxor.HackerNewsApi`.

        :type QUERY_UNSEEN: str (const)
        :param foo: the query to show unseen comments.

        :type web_viewer: :class:`web_viewer.WebViewer`
        :param web_viewer: An instance of `web_viewer.WebViewer`.
    """
    COMMENT_INDENT = '  '
    COMMENT_UNSEEN = ' [!]'
    MAX_LIST_INDEX = 1000
    MAX_SNIPPET_LENGTH = 60
    QUERY_UNSEEN = '\\[!\\]'

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.hacker_news_api = HackerNewsApi()
        try:
            self.html = HTMLParser.HTMLParser()
        except:
            self.html = HTMLParser
        self.config = Config()
        self.web_viewer = WebViewer()

    def ask(self, limit):
        if False:
            while True:
                i = 10
        'Display Ask HN posts.\n\n        :type limit: int\n        :param limit: the number of items to show, optional, defaults to 10.\n        '
        self.print_items(message=self.headlines_message('Ask HN'), item_ids=self.hacker_news_api.ask_stories(limit))

    def best(self, limit):
        if False:
            print('Hello World!')
        'Display best posts.\n\n        :type limit: int\n        :param limit: the number of items to show, optional, defaults to 10.\n        '
        self.print_items(message=self.headlines_message('Best'), item_ids=self.hacker_news_api.best_stories(limit))

    def headlines_message(self, message):
        if False:
            while True:
                i = 10
        'Create the "Fetching [message] Headlines..." string.\n\n        :type message: str\n        :param message: The headline message.\n\n        :rtype: str\n        :return: "Fetching [message] Headlines...".\n        '
        return 'Fetching {0} Headlines...'.format(message)

    def hiring_and_freelance(self, regex_query, post_id):
        if False:
            while True:
                i = 10
        'Display comments matching the monthly who is hiring post.\n\n        Searches the monthly Hacker News who is hiring post for comments\n        matching the given regex_query.  Defaults to searching the latest\n        post based on your installed version of haxor-news.\n\n        :type regex_query: str\n        :param regex_query: The regex query to match.\n\n        :type post_id: int\n        :param post_id: the who is hiring post id.\n                Optional, defaults to the latest post based on your installed\n                version of haxor-news.\n        '
        try:
            item = self.hacker_news_api.get_item(post_id)
            self.print_comments(item, regex_query, comments_hide_non_matching=True)
            self.config.save_cache()
        except InvalidItemID:
            self.print_item_not_found(post_id)
        except IOError:
            sys.stderr.close()

    def jobs(self, limit):
        if False:
            while True:
                i = 10
        'Display job posts.\n\n        :type limit: int\n        :param limit: the number of items to show, optional, defaults to 10.\n        '
        self.print_items(message=self.headlines_message('Jobs'), item_ids=self.hacker_news_api.job_stories(limit))

    def new(self, limit):
        if False:
            return 10
        'Display the latest posts.\n\n        :type limit: int\n        :param limit: the number of items to show, optional, defaults to 10.\n        '
        self.print_items(message=self.headlines_message('Latest'), item_ids=self.hacker_news_api.new_stories(limit))

    def onion(self, limit):
        if False:
            print('Hello World!')
        'Display onions.\n\n        :type limit: int\n        :param limit: the number of items to show, optional, defaults to 10.\n        '
        click.secho('\n{h}\n'.format(h=self.headlines_message('Top Onion')), fg=self.config.clr_title)
        index = 1
        for onion in onions[0:limit]:
            formatted_index_title = self.format_index_title(index, onion)
            click.echo(formatted_index_title)
            index += 1
        click.echo('')

    def print_comment(self, item, regex_query='', comments_hide_non_matching=False, depth=0):
        if False:
            while True:
                i = 10
        "Print the comments for the given item.\n\n        :type item: :class:`haxor.Item`\n        :param item: An instance of `haxor.Item`.\n\n        :type regex_query: str\n        :param regex_query: the regex query to match.\n\n        :type comments_hide_non_matching: bool\n        :param comments_hide_non_matching: determines whether to\n                hide comments that don't match (False) or truncate them (True).\n\n        :type depth: int\n        :param depth: The current recursion depth, used to indent the comment.\n        "
        if item.text is None:
            return
        header_color = 'yellow'
        header_color_highlight = 'magenta'
        header_adornment = ''
        if self.config.item_cache is not None and str(item.item_id) not in self.config.item_cache:
            header_adornment = self.COMMENT_UNSEEN
            self.config.item_cache.append(item.item_id)
        show_comment = True
        if regex_query is not None:
            if self.match_comment_unseen(regex_query, header_adornment) or self.match_regex(item, regex_query):
                header_color = header_color_highlight
            else:
                show_comment = False
        (formatted_heading, formatted_comment) = self.format_comment(item, depth, header_color, header_adornment)
        if show_comment:
            click.echo(formatted_heading, color=True)
            click.echo(formatted_comment, color=True)
        elif comments_hide_non_matching:
            click.secho('.', nl=False)
        else:
            click.echo(formatted_heading, color=True)
            num_chars = len(formatted_comment)
            if num_chars > self.MAX_SNIPPET_LENGTH:
                num_chars = self.MAX_SNIPPET_LENGTH
            click.echo(formatted_comment[0:num_chars] + ' [...]', color=True)

    def print_comments(self, item, regex_query='', comments_hide_non_matching=False, depth=0):
        if False:
            return 10
        "Recursively print comments and subcomments for the given item.\n\n        :type item: :class:`haxor.Item`\n        :param item: An instance of `haxor.Item`.\n\n        :type regex_query: str\n        :param regex_query: the regex query to match.\n\n        :type comments_hide_non_matching: bool\n        :param comments_hide_non_matching: determines whether to\n                hide comments that don't match (False) or truncate them (True).\n\n        :type depth: int\n        :param depth: The current recursion depth, used to indent the comment.\n        "
        self.print_comment(item, regex_query, comments_hide_non_matching, depth)
        comment_ids = item.kids
        if not comment_ids:
            return
        for comment_id in comment_ids:
            try:
                comment = self.hacker_news_api.get_item(comment_id)
                depth += 1
                self.print_comments(comment, regex_query=regex_query, comments_hide_non_matching=comments_hide_non_matching, depth=depth)
                depth -= 1
            except (InvalidItemID, HTTPError):
                click.echo('')
                self.print_item_not_found(comment_id)

    def format_comment(self, item, depth, header_color, header_adornment):
        if False:
            for i in range(10):
                print('nop')
        "Format a given item's comment.\n\n        :type item: :class:`haxor.Item`\n        :param item: An instance of `haxor.Item`.\n\n        :type depth: int\n        :param depth: The current recursion depth, used to indent the comment.\n\n        :type header_color: str\n        :param header_color: The header color.\n\n        :type header_adornment: str\n        :param header_adornment: The header adornment.\n\n        :rtype: tuple\n        :return: * A string representing the formatted comment header.\n                 * A string representing the formatted comment.\n        "
        indent = self.COMMENT_INDENT * depth
        formatted_heading = click.style('\n{i}{b} - {d}{h}'.format(i=indent, b=item.by, d=str(pretty_date_time(item.submission_time)), h=header_adornment), fg=header_color)
        unescaped_text = self.html.unescape(item.text)
        regex_paragraph = re.compile('<p>')
        unescaped_text = regex_paragraph.sub(click.style('\n\n' + indent), unescaped_text)
        regex_url = re.compile('(<a href=(".*") .*</a>)')
        unescaped_text = regex_url.sub(click.style('\\2', fg=self.config.clr_link), unescaped_text)
        regex_tag = re.compile('(<(.*)>.*?<\\/\\2>)')
        unescaped_text = regex_tag.sub(click.style('\\1', fg=self.config.clr_tag), unescaped_text)
        formatted_comment = click.wrap_text(text=unescaped_text, initial_indent=indent, subsequent_indent=indent)
        return (formatted_heading, formatted_comment)

    def format_index_title(self, index, title):
        if False:
            i = 10
            return i + 15
        "Format and item's index and title.\n\n        :type index: int\n        :param index: The index for the given item, used with the\n            hn view [index] commend.\n\n        :type title: str\n        :param title: The item's title.\n\n        :rtype: str\n        :return: The formatted index and title.\n        "
        INDEX_PAD = 5
        formatted_index = '  ' + (str(index) + '.').ljust(INDEX_PAD)
        formatted_index_title = click.style(formatted_index, fg=self.config.clr_view_index)
        formatted_index_title += click.style(title + ' ', fg=self.config.clr_title)
        return formatted_index_title

    def format_item(self, item, index):
        if False:
            i = 10
            return i + 15
        'Format an item.\n\n        :type item: :class:`haxor.Item`\n        :param item: An instance of `haxor.Item`.\n\n        :type index: int\n        :param index: The index for the given item, used with the\n            hn view [index] commend.\n\n        :rtype: str\n        :return: The formatted item.\n        '
        formatted_item = self.format_index_title(index, item.title)
        if item.url is not None:
            netloc = urlparse(item.url).netloc
            netloc = re.sub('www.', '', netloc)
            formatted_item += click.style('(' + netloc + ')', fg=self.config.clr_view_link)
        formatted_item += '\n         '
        formatted_item += click.style(str(item.score) + ' points ', fg=self.config.clr_num_points)
        formatted_item += click.style('by ' + item.by + ' ', fg=self.config.clr_user)
        submission_time = str(pretty_date_time(item.submission_time))
        formatted_item += click.style(submission_time + ' ', fg=self.config.clr_time)
        num_comments = str(item.descendants) if item.descendants else '0'
        formatted_item += click.style('| ' + num_comments + ' comments', fg=self.config.clr_num_comments)
        return formatted_item

    def print_item_not_found(self, item_id):
        if False:
            for i in range(10):
                print('nop')
        "Print a message the given item id was not found.\n\n        :type item_id: int\n        :param item_id: The item's id.\n        "
        click.secho('Item with id {0} not found.'.format(item_id), fg='red')

    def print_items(self, message, item_ids):
        if False:
            for i in range(10):
                print('nop')
        'Print the items.\n\n        :type message: str\n        :param message: A message to print out to the user before outputting\n                the results.\n\n        :type item_ids: iterable\n        :param item_ids: A collection of items to print.\n                Can be a list or dictionary.\n        '
        self.config.item_ids = []
        index = 1
        for item_id in item_ids:
            try:
                item = self.hacker_news_api.get_item(item_id)
                if item.title:
                    formatted_item = self.format_item(item, index)
                    self.config.item_ids.append(item.item_id)
                    click.echo(formatted_item)
                    index += 1
            except InvalidItemID:
                self.print_item_not_found(item_id)
        self.config.save_cache()
        if self.config.show_tip:
            click.secho(self.tip_view(str(index - 1)))

    def tip_view(self, max_index):
        if False:
            for i in range(10):
                print('nop')
        'Create the tip about the view command.\n\n        :type max_index: string\n        :param max_index: The index uppor bound, used with the\n            hn view [index] commend.\n\n        :rtype: str\n        :return: The formatted tip.\n        '
        tip = click.style('  Tip: View the page or comments for ', fg=self.config.clr_tooltip)
        tip += click.style('1 through ', fg=self.config.clr_view_index)
        tip += click.style(str(max_index), fg=self.config.clr_view_index)
        tip += click.style(' with the following command:\n', fg=self.config.clr_tooltip)
        tip += click.style('    hn view [#] ', fg=self.config.clr_view_index)
        tip += click.style('optional: [-c] [-cr] [-cu] [-cq "regex"] [-ch] [-b] [--help]' + '\n', fg=self.config.clr_tooltip)
        return tip

    def match_comment_unseen(self, regex_query, header_adornment):
        if False:
            for i in range(10):
                print('nop')
        'Determine if a comment is unseen based on the query and header.\n\n        :type regex_query: str\n        :param regex_query: The regex query to match.\n\n        :type header_adornment: str\n        :param header_adornment: The header adornment.\n\n        :rtype: bool\n        :return: Specifies if there is a match found.\n        '
        if regex_query == self.QUERY_UNSEEN and header_adornment == self.COMMENT_UNSEEN:
            return True
        else:
            return False

    def match_regex(self, item, regex_query):
        if False:
            i = 10
            return i + 15
        'Determine if there is a match with the given regex_query.\n\n        :type item: :class:`haxor.Item`\n        :param item: An instance of `haxor.Item`.\n\n        :type regex_query: str\n        :param regex_query: The regex query to match.\n\n        :rtype: bool\n        :return: Specifies if there is a match found.\n        '
        match_time = re.search(regex_query, str(pretty_date_time(item.submission_time)))
        match_user = re.search(regex_query, item.by)
        match_text = re.search(regex_query, item.text)
        if not match_text and (not match_user) and (not match_time):
            return False
        else:
            return True

    def show(self, limit):
        if False:
            i = 10
            return i + 15
        'Display Show HN posts.\n\n        :type limit: int\n        :param limit: the number of items to show, optional, defaults to 10.\n        '
        self.print_items(message=self.headlines_message('Show HN'), item_ids=self.hacker_news_api.show_stories(limit))

    def top(self, limit):
        if False:
            print('Hello World!')
        'Display the top posts.\n\n        :type limit: int\n        :param limit: the number of items to show, optional, defaults to 10.\n        '
        self.print_items(message=self.headlines_message('Top'), item_ids=self.hacker_news_api.top_stories(limit))

    def user(self, user_id, submission_limit):
        if False:
            for i in range(10):
                print('nop')
        "Display basic user info and submitted posts.\n\n        :type user_id: str.\n        :param user_id: The user'd login name.\n\n        :type submission_limit: int\n        :param submission_limit: the number of submissions to show.\n                Optional, defaults to 10.\n        "
        try:
            user = self.hacker_news_api.get_user(user_id)
            click.secho('\nUser Id: ', nl=False, fg=self.config.clr_general)
            click.secho(user_id, fg=self.config.clr_user)
            click.secho('Created: ', nl=False, fg=self.config.clr_general)
            click.secho(str(user.created), fg=self.config.clr_user)
            click.secho('Karma: ', nl=False, fg=self.config.clr_general)
            click.secho(str(user.karma), fg=self.config.clr_user)
            self.print_items('User submissions:', user.submitted[0:submission_limit])
        except InvalidUserID:
            self.print_item_not_found(user_id)

    def view(self, index, comments_query, comments, comments_hide_non_matching, browser):
        if False:
            while True:
                i = 10
        "View the given index contents.\n\n        Uses ids from ~/.haxornewsconfig stored in self.config.item_ids.\n        If url is True, opens a browser with the url based on the given index.\n        Else, displays the post's comments.\n\n        :type index: int\n        :param index: The index for the given item, used with the\n            hn view [index] commend.\n\n        :type comments: bool\n        :param comments: Determines whether to view the comments\n                or a simplified version of the post url.\n\n        :type comments_hide_non_matching: bool\n        :param comments_hide_non_matching: determines whether to\n                hide comments that don't match (False) or truncate them (True).\n\n        :type browser: bool\n        :param browser: determines whether to view the url in a browser.\n        "
        if self.config.item_ids is None:
            click.secho('There are no posts indexed, run a command such as hn top first', fg='red')
            return
        item_id = index
        if index < self.MAX_LIST_INDEX:
            try:
                item_id = self.config.item_ids[index - 1]
            except IndexError:
                self.print_item_not_found(item_id)
                return
        try:
            item = self.hacker_news_api.get_item(item_id)
        except InvalidItemID:
            self.print_item_not_found(self.config.item_ids[index - 1])
            return
        if not comments and item.url is None:
            click.secho('\nNo url associated with post.', nl=False, fg=self.config.clr_general)
            comments = True
        if comments:
            comments_url = 'https://news.ycombinator.com/item?id=' + str(item.item_id)
            click.secho('\nFetching Comments from ' + comments_url, fg=self.config.clr_general)
            if browser:
                webbrowser.open(comments_url)
            else:
                try:
                    self.print_comments(item, regex_query=comments_query, comments_hide_non_matching=comments_hide_non_matching)
                    click.echo('')
                except IOError:
                    sys.stderr.close()
                self.config.save_cache()
        else:
            click.secho('\nOpening ' + item.url + ' ...', fg=self.config.clr_general)
            if browser:
                webbrowser.open(item.url)
            else:
                contents = self.web_viewer.generate_url_contents(item.url)
                header = click.style('Viewing ' + item.url + '\n\n', fg=self.config.clr_general)
                contents = header + contents
                contents += click.style('\nView this article in a browser with the -b/--browser flag.\n', fg=self.config.clr_general)
                contents += click.style('\nPress q to quit viewing this article.\n', fg=self.config.clr_general)
                if platform.system() == 'Windows':
                    try:
                        contents = re.sub('[^\\x00-\\x7F]+', '', contents)
                        click.echo(contents)
                    except IOError:
                        sys.stderr.close()
                else:
                    click.echo_via_pager(contents)
            click.echo('')

    def view_setup(self, index, comments_regex_query, comments, comments_recent, comments_unseen, comments_hide_non_matching, clear_cache, browser):
        if False:
            i = 10
            return i + 15
        "Set up the call to view the given index comments or url.\n\n        This method is meant to be called after a command that outputs a\n        table of posts.\n\n        :type index: int\n        :param index: The index for the given item, used with the\n            hn view [index] commend.\n\n        :type regex_query: str\n        :param regex_query: The regex query to match.\n\n        :type comments: bool\n        :param comments: Determines whether to view the comments\n                or a simplified version of the post url.\n\n        :type comments_recent: bool\n        :param comments_recent: Determines whether to view only\n                recently comments (posted within the past 59 minutes or less).\n\n        :type comments_unseen: bool\n        :param comments_unseen: Determines whether to view only\n                comments that you have not yet seen.\n\n        :type comments_hide_non_matching: bool\n        :param comments_hide_non_matching: determines whether to\n                hide comments that don't match (False) or truncate them (True).\n\n        :type clear_cache: bool\n        :param clear_cache: foos.\n\n        :type browser: bool\n        :param browser: Determines whether to clear the comment cache before\n            running the view command.\n        "
        if comments_regex_query is not None:
            comments = True
        if comments_recent:
            comments_regex_query = 'seconds ago|minutes ago'
            comments = True
        if comments_unseen:
            comments_regex_query = self.QUERY_UNSEEN
            comments = True
        if clear_cache:
            self.config.clear_item_cache()
        self.view(int(index), comments_regex_query, comments, comments_hide_non_matching, browser)